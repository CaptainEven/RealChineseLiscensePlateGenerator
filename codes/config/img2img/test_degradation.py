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


def generate_LR_HR_pairs(model,
                         src_dir,
                         dst_dir,
                         ext=".jpg",
                         down_scale=2):
    """
    @param model:
    @param src_dir:
    @param dst_dir:
    @param ext:
    @param down_scale:
    @return:
    """
    src_dir = os.path.abspath(src_dir)
    if not os.path.isdir(src_dir):
        print("[Err]: invalid src dir: {:s}, exit now!"
              .format(src_dir))
        exit(-1)

    # dst_dir = os.path.abspath(dst_dir)
    # if os.path.isdir(dst_dir):
    #     shutil.rmtree(dst_dir)
    # try:
    #     os.makedirs(dst_dir)
    # except Exception as e:
    #     print(e)
    #     exit(-1)

    dst_HR_dir = os.path.abspath(dst_dir + "/HR")
    if not os.path.isdir(dst_HR_dir):
        os.makedirs(dst_HR_dir)
    dst_LR_dir = os.path.abspath(dst_dir + "/LR")
    if not os.path.isdir(dst_LR_dir):
        os.makedirs(dst_LR_dir)
    dst_LR_sub_dir = os.path.abspath(dst_LR_dir + "/X{:d}"
                                     .format(int(down_scale)))
    if not os.path.isdir(dst_LR_sub_dir):
        os.makedirs(dst_LR_sub_dir)

    f_paths = []
    find_files(src_dir, f_paths, ext)
    print("\n[Info]: find total {:d} files of [{:s}]\n"
          .format(len(f_paths), ext))
    with tqdm(total=len(f_paths)) as p_bar:
        for f_path in f_paths:
            f_name = os.path.split(f_path)[-1]

            # ----- generate HR image
            dst_hr_path = os.path.abspath(dst_HR_dir + "/" + f_name)
            if not os.path.isfile(dst_hr_path):
                shutil.copy(f_path, dst_HR_dir)  # copy HR img
                print("\b--> {:s} [cp to] {:s}\n".
                      format(f_name, dst_HR_dir))

            # ----- generate LR image
            hr = cv2.imread(f_path, cv2.IMREAD_COLOR)
            h, w, c = hr.shape
            lr = cv2.resize(hr, (w // down_scale, h // down_scale), cv2.INTER_CUBIC)
            dst_lr_path = os.path.abspath(dst_LR_sub_dir + "/" + f_name)
            if not os.path.isfile(dst_lr_path):
                LQ = util.img2tensor(lr)
                LQ = LQ.unsqueeze(0)  # CHW -> NCHW

                # ---------- Inference
                noisy_state = sde.noise_state(LQ)
                model.feed_data(noisy_state, LQ)
                model.test(sde, save_states=True)

                # ---------- Get output
                visuals = model.get_current_visuals(need_GT=False)  # gpu -> cpu
                output = visuals["Output"]
                HQ = util.tensor2img(output.squeeze())  # uint8

                # ----- Save output
                util.save_img_uncompressed(dst_lr_path, HQ)  # save LR img
                print("\n--> {:s} [generated at] {:s}\n"
                      .format(f_name, dst_LR_sub_dir))
            p_bar.update()


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
        h, w, c = img.shape
        print("--> degradating {:s} of input size {:d}×{:d}..."
              .format(img_name, w, h))

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


if __name__ == "__main__":
    # test_text2img(args, model, sde)
    # run_degradation(model,
    #                 src_img_dir="/mnt/diske/ROIs",
    #                 dst_img_dir="/mnt/diske/lyw/Degradations",
    #                 ext=".jpg")

    generate_LR_HR_pairs(model,
                         src_dir="/mnt/diske/RandomSamples",
                         dst_dir="/mnt/ssd/lyw/SISR_data",
                         ext=".jpg")
