# encoding=utf-8

import argparse
import logging
import os.path
import shutil
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils
from easydict import EasyDict as edict
import numpy as np
import torch
from IPython import embed
import lpips

import options as option
from the_models import create_model

sys.path.insert(0, "../../")
import codes.sdeutils as util
from codes.data import create_dataloader, create_dataset
from codes.data.util import bgr2ycbcr

import cv2
from LicensePlateGenerator.generate_multi_plate import MultiPlateGenerator
from LicensePlateGenerator.generate_special_plate import generate_one_plate

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt",
                    type=str,
                    default="./options/test/ir-sde.yml",
                    help="Path to options YMAL file.")
parser.add_argument("--text",
                    type=str,
                    default="使014578_black_single",
                    help="")
parser.add_argument("--list_file",
                    type=str,
                    default="./plates.txt",
                    help="")
parser.add_argument("--type",
                    type=str,
                    default="list",
                    help="single | list")

args = parser.parse_args()
args = edict(vars(args))  # vars()函数返回对象object的属性和属性值的字典对象。

opt = option.parse(args.opt, is_train=False)

opt = option.dict_to_nonedict(opt)
args.opt = opt

#### mkdir and logger
util.mkdirs((path
             for key, path in opt["path"].items()
             if not key == "experiments_root"
             and "pretrain_model" not in key
             and "resume" not in key))

os.system("rm ./result")
os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

util.setup_logger("base",
                  opt["path"]["log"],
                  "test_" + opt["name"],
                  level=logging.INFO,
                  screen=True,
                  tofile=True, )
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info("[Info]: number of test images in [{:s}]: {:d}"
                .format(dataset_opt["name"],
                        len(test_set)))
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
device = model.device

sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"],
                 T=opt["sde"]["T"],
                 schedule=opt["sde"]["schedule"],
                 eps=opt["sde"]["eps"],
                 device=device)
sde.set_model(model.model)

lpips_fn = lpips.LPIPS(net='alex').to(device)
scale = opt['degradation']['scale']


def run_test_set():
    """
    @return:
    """
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt["name"]  # path opt['']
        logger.info("\nTesting [{:s}]...".format(test_set_name))
        test_start_time = time.time()
        dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
        util.mkdir(dataset_dir)

        test_results = OrderedDict()
        test_results["psnr"] = []
        test_results["ssim"] = []
        test_results["psnr_y"] = []
        test_results["ssim_y"] = []
        test_results["lpips"] = []
        test_times = []

        for i, test_data in enumerate(test_loader):
            single_img_psnr = []
            single_img_ssim = []
            single_img_psnr_y = []
            single_img_ssim_y = []
            need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
            img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            #### input dataset_LQ
            LQ, GT = test_data["LQ"], test_data["GT"]
            noisy_state = sde.noise_state(LQ)

            model.feed_data(noisy_state, LQ, GT)
            tic = time.time()
            model.test(sde, save_states=True)
            toc = time.time()
            test_times.append(toc - tic)

            visuals = model.get_current_visuals()
            SR_img = visuals["Output"]
            output = util.tensor2img(SR_img.squeeze())  # uint8
            LQ_ = util.tensor2img(visuals["Input"].squeeze())  # uint8
            GT_ = util.tensor2img(visuals["GT"].squeeze())  # uint8

            suffix = opt["suffix"]
            if suffix:
                save_img_path = os.path.join(dataset_dir, img_name + suffix + ".png")
            else:
                save_img_path = os.path.join(dataset_dir, img_name + "_GEN.png")
            util.save_img(output, save_img_path)

            # remove it if you only want to save output images
            LQ_img_path = os.path.join(dataset_dir, img_name + "_LQ.png")
            GT_img_path = os.path.join(dataset_dir, img_name + "_GT.png")
            util.save_img(LQ_, LQ_img_path)
            util.save_img(GT_, GT_img_path)

            if need_GT:
                gt_img = GT_ / 255.0
                sr_img = output / 255.0

                crop_border = opt["crop_border"] if opt["crop_border"] else scale
                if crop_border == 0:
                    cropped_sr_img = sr_img
                    cropped_gt_img = gt_img
                else:
                    cropped_sr_img = sr_img[
                                     crop_border:-crop_border, crop_border:-crop_border
                                     ]
                    cropped_gt_img = gt_img[
                                     crop_border:-crop_border, crop_border:-crop_border
                                     ]

                psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
                lp_score = lpips_fn(GT.to(device) * 2 - 1, SR_img.to(device) * 2 - 1).squeeze().item()

                test_results["psnr"].append(psnr)
                test_results["ssim"].append(ssim)
                test_results["lpips"].append(lp_score)

                if len(gt_img.shape) == 3:
                    if gt_img.shape[2] == 3:  # RGB image
                        sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                        gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                        if crop_border == 0:
                            cropped_sr_img_y = sr_img_y
                            cropped_gt_img_y = gt_img_y
                        else:
                            cropped_sr_img_y = sr_img_y[crop_border:-crop_border,
                                               crop_border:-crop_border]
                            cropped_gt_img_y = gt_img_y[crop_border:-crop_border,
                                               crop_border:-crop_border]
                        psnr_y = util.calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                        ssim_y = util.calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)

                        test_results["psnr_y"].append(psnr_y)
                        test_results["ssim_y"].append(ssim_y)

                        logger.info("img{:3d}:{:15s} - PSNR: {:.6f} dB;"
                                    " SSIM: {:.6f}; LPIPS: {:.6f};"
                                    " PSNR_Y: {:.6f} dB; "
                                    "SSIM_Y: {:.6f}.".format(i,
                                                             img_name,
                                                             psnr,
                                                             ssim,
                                                             lp_score,
                                                             psnr_y,
                                                             ssim_y))
                else:
                    logger.info("img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}."
                                .format(img_name, psnr, ssim))

                    test_results["psnr_y"].append(psnr)
                    test_results["ssim_y"].append(ssim)
            else:
                logger.info(img_name)

        ave_lpips = sum(test_results["lpips"]) / len(test_results["lpips"])
        ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
        ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
        logger.info("----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB;"
                    " SSIM: {:.6f}\n".format(test_set_name, ave_psnr, ave_ssim))

        if test_results["psnr_y"] and test_results["ssim_y"]:
            ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
            ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])
            logger.info("----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; "
                        "SSIM_Y: {:.6f}\n".format(ave_psnr_y, ave_ssim_y))

        logger.info("----average LPIPS\t: {:.6f}\n".format(ave_lpips))

        print(f"average test time: {np.mean(test_times):.4f}")


def text2img(txt, model, generator, dataset_dir, n_gen=10):
    """
    @param txt:
    @param model:
    @param generator:
    @param dataset_dir:
    @param n_gen:
    @return:
    """
    # ---------- Generate standard license plate img
    fields = txt.split("_")
    assert len(fields) >= 3
    plate_number = fields[0]
    plate_color = fields[1]
    plate_layers = fields[2]
    is_double = plate_layers == "double"
    img_name = txt
    LQ = generator.generate_plate_special(plate_number=plate_number,
                                          bg_color=plate_color,
                                          is_double=is_double)
    model_img_path = "/mnt/diske/{:s}.png".format(plate_number)
    cv2.imwrite(model_img_path, LQ)
    print("[Info]: {:s} saved".format(model_img_path))

    # ---------- Generate HQ img
    if plate_layers == "single":
        h, w, c = LQ.shape
        if w != 192 or h != 64:
            LQ = cv2.resize(LQ, (192, 64), cv2.INTER_LINEAR)  # BGR
            LQ = LQ.astype(np.float32) / 255.0  # normalize to [0, 1]

            # ---------- BGR to RGB, HWC to CHW, numpy to tensor
            if LQ.shape[2] == 3:
                LQ = LQ[:, :, [2, 1, 0]]  # BGR2RGB
            LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(LQ, (2, 0, 1)))).float()
            LQ = LQ.unsqueeze(0)  # CHW -> NCHW

            for i in range(n_gen):
                # ---------- Inference
                noisy_state = sde.noise_state(LQ)
                model.feed_data(noisy_state, LQ)
                model.test(sde, save_states=True)

                # ---------- Get output
                visuals = model.get_current_visuals(need_GT=False)
                output = visuals["Output"]
                output = util.tensor2img(output.squeeze())  # uint8

                # ---------- Save output
                save_img_path = dataset_dir + "/" \
                                + img_name + "_GEN_{:d}.png".format(i + 1)
                save_img_path = os.path.abspath(save_img_path)
                cv2.imwrite(save_img_path, output)
                print("--> {:s} generated".format(save_img_path))


def test_text2img(args, model, sde):
    """
    @param args:
    @param model:
    @param sde:
    @return:
    """
    opt = args.opt

    # ---------- Set dataset dir path
    test_set_name = test_loader.dataset.opt["name"]
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    dataset_dir = os.path.abspath(dataset_dir)
    if os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)
    util.mkdir(dataset_dir)

    # ---------- Define a generator
    generator = MultiPlateGenerator('../../LicensePlateGenerator/plate_model',
                                    '../../LicensePlateGenerator/font_model')

    if args.type == "single":
        print("[Info]: generation from single...")
        text2img(args.text, model, generator, dataset_dir, n_gen=10)
    elif args.type == "list":
        print("[Info]: generation from file list...")
        list_file_path = os.path.abspath(args.list_file)
        if not os.path.isfile(list_file_path):
            print("[Err]: invalid list file: {:s}"
                  .format(list_file_path))
            exit(-1)
        with open(list_file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                text = line.strip()
                print("--> generating {:s}...".format(text))
                text2img(text, model, generator, dataset_dir, n_gen=10)


if __name__ == "__main__":
    test_text2img(args, model, sde)
