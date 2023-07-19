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
from tqdm import tqdm

to_be_inserted_path = os.path.abspath("../../")
sys.path.insert(0, to_be_inserted_path)
print("[Info]: {:s} inserted".format(to_be_inserted_path))

to_be_inserted_path = os.path.abspath("../../../")
sys.path.insert(0, to_be_inserted_path)
print("[Info]: {:s} inserted".format(to_be_inserted_path))

import codes.config.img2img.options as option
from the_models import create_model

import codes.sdeutils as util
from codes.data import create_dataloader, create_dataset
from codes.data.util import bgr2ycbcr
from codes.sdeutils.img_utils import calculate_psnr, calculate_ssim, rmse, mse

import cv2
from LicensePlateGenerator.generate_multi_plate import MultiPlateGenerator
from LicensePlateGenerator.generate_special_plate import generate_one_plate

from gen_random_plate_string import generate_random_plate_text

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt",
                    type=str,
                    default="./options/test/degradation.yml",
                    help="Path to options YMAL file.")
parser.add_argument("-s",
                    "--src_img_dir",
                    type=str,
                    default="",
                    help="")

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


def find_files(root, f_list, ext):
    """
    Find all files with an extension
    Args:
        root:
        f_list:
        ext:
    Returns:
    """
    for f_name in os.listdir(root):
        f_path = os.path.join(root, f_name)
        if os.path.isfile(f_path) and f_name.endswith(ext):
            f_list.append(f_path)
        elif os.path.isdir(f_path):
            find_files(f_path, f_list, ext)


def run_degradation(model,
                    src_img_dir,
                    dst_img_dir,
                    ext=".jpg"):
    """
    @param model:
    @param src_img_dir:
    @param dst_img_dir:
    @param ext:
    @return:
    """
    src_img_dir = os.path.abspath(src_img_dir)
    if not os.path.isdir(src_img_dir):
        print("[Err]: invalid src img dir: {:s}"
              .format(src_img_dir))
        exit(-1)

    dst_img_dir = os.path.abspath(dst_img_dir)
    if os.path.isdir(dst_img_dir):
        shutil.rmtree(dst_img_dir)
    try:
        os.makedirs(dst_img_dir)
    except Exception as e:
        print(e)
    else:
        print("[Info]: {:s} made".format(dst_img_dir))

    img_paths = []
    find_files(src_img_dir, img_paths, ext)
    print("[Info]: find total {:d} imgs of [{:s}] files"
          .format(len(img_paths), ext))
    for img_path in img_paths:
        img_name = os.path.split(img_path)[-1]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        print("--> degradate {:s}...".format(img_name))
        h, w, c = img.shape

        # normalize to [0, 1]
        LQ = img.astype(np.float32) / 255.0

        # BGR to RGB, HWC to CHW, numpy to tensor
        if LQ.shape[2] == 3:
            LQ = LQ[:, :, [2, 1, 0]]  # BGR2RGB
        LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(LQ, (2, 0, 1)))).float()
        LQ = LQ.unsqueeze(0)  # CHW -> NCHW

        # ---------- Inference
        noisy_state = sde.noise_state(LQ)
        model.feed_data(noisy_state, LQ)
        model.test(sde, save_states=True)

        # ---------- Get output
        visuals = model.get_current_visuals(need_GT=False)  # gpu -> cpu
        output = visuals["Output"]
        HQ = util.tensor2img(output.squeeze())  # uint8

        # ---------- Save output
        dst_save_path = os.path.abspath(dst_img_dir + "/" + img_name)
        if not os.path.isfile(dst_save_path):
            # cv2.imwrite(dst_save_path, HQ)
            if img_name.endswith(".jpg"):
                cv2.imencode(".jpg", HQ, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tofile(dst_save_path)
            elif img_name.endswith(".png"):
                cv2.imencode(".png", HQ, [cv2.IMWRITE_PNG_COMPRESSION, 0])[1].tofile(dst_save_path)
            else:
                print("[Warning]: invalid file type: {:s}".format(img_name.split(".")[-1]))
                continue
            print("--> {:s} saved".format(dst_save_path))


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
    LQ_np_bgr = generator.generate_plate_special(plate_number=plate_number,
                                                 bg_color=plate_color,
                                                 is_double=is_double)
    model_img_path = "/mnt/diske/{:s}.png".format(plate_number)
    cv2.imwrite(model_img_path, LQ_np_bgr)
    # print("[Info]: model image(LQ) {:s} saved".format(model_img_path))

    # ---------- Generate HQ img
    if plate_layers == "single":  # single layer license plate image
        h, w, c = LQ_np_bgr.shape
        if w != 192 or h != 64:
            LQ_np_bgr = cv2.resize(LQ_np_bgr, (192, 64), cv2.INTER_LINEAR)  # BGR

        # normalize to [0, 1]
        LQ = LQ_np_bgr.astype(np.float32) / 255.0

        # ---------- BGR to RGB, HWC to CHW, numpy to tensor
        if LQ.shape[2] == 3:
            LQ = LQ[:, :, [2, 1, 0]]  # BGR2RGB
        LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(LQ, (2, 0, 1)))).float()
        LQ = LQ.unsqueeze(0)  # CHW -> NCHW
        with tqdm(total=n_gen) as p_bar:
            for i in range(n_gen):
                # ---------- Inference
                noisy_state = sde.noise_state(LQ)
                model.feed_data(noisy_state, LQ)
                model.test(sde, save_states=True)

                # ---------- Get output
                visuals = model.get_current_visuals(need_GT=False)  # gpu -> cpu
                output = visuals["Output"]
                HQ = util.tensor2img(output.squeeze())  # uint8

                # # ---------- Calculate LQ/HQ similarity
                # ssim_val = calculate_ssim(LQ_np_bgr, HQ)
                # psnr_val = calculate_psnr(LQ_np_bgr, HQ)
                # rmse_val = rmse(LQ_np_bgr, HQ)

                # ---------- Save output
                save_img_path = dataset_dir + "/" \
                                + img_name + "_GEN_{:d}.png" \
                                    .format(i + 1)
                # save_img_path = dataset_dir + "/" \
                #                 + img_name + "_GEN_{:d}_ssim{:.3f}.png" \
                #                     .format(i + 1, ssim_val)
                save_img_path = os.path.abspath(save_img_path)
                cv2.imwrite(save_img_path, HQ)
                # print("\n--> {:s} generated\n".format(save_img_path))
                p_bar.update()
    elif plate_layers == "double":
        print("[Warning]: double not surported now!")
        return


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
        print("[Info]: single mode")
        text2img(args.text, model, generator, dataset_dir, n_gen=10)
    elif args.type == "list":
        print("[Info]: list mode")
        print("[Info]: generation from file list...")
        list_file_path = os.path.abspath(args.list_file)
        if not os.path.isfile(list_file_path):
            print("[Err]: invalid list file: {:s}"
                  .format(list_file_path))
            exit(-1)

        n_files = 0
        with open(list_file_path, "r", encoding="utf-8") as f:
            n_files = len(f.readlines())
        print("\n[Info]: total {:d} files to be generated".format(n_files))
        with open(list_file_path, "r", encoding="utf-8") as f:
            file_i = 0
            for line in f.readlines():
                file_i += 1
                text = line.strip()
                print("\n--> generating {:s}, ({:3d}/{:3d})...\n"
                      .format(text, file_i, n_files))
                text2img(text, model, generator, dataset_dir, n_gen=10)
    elif args.type == "instant":
        print("[Info]: instant mode")
        color = ["blue", "yellow", "white", "black", "green"]
        # layer = ["single", "double"]
        layer = ["single"]
        yellow_special = ["xue", "gua", "normal"]
        white_special = ["wujing", "jun", "jing", "yingji"]
        green_special = ["bus", "normal"]
        black_special = ["gang", "ao", "dashi", "lingshi"]
        for i in range(args.num):
            text = generate_random_plate_text(color,
                                              layer,
                                              yellow_special,
                                              white_special,
                                              green_special,
                                              black_special)
            print("--> generating {:s}, {:3d}/{:3d}..."
                  .format(text, i + 1, args.num))
            text2img(text, model, generator, dataset_dir, n_gen=3)


if __name__ == "__main__":
    # test_text2img(args, model, sde)
    run_degradation(model,
                    src_img_dir="/mnt/diske/ROIs",
                    dst_img_dir="/mnt/diske/lyw/Degradations",
                    ext=".png")
