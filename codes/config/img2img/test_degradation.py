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
import time
from IPython import embed
import lpips
from tqdm import tqdm
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

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


def get_model(opt, is_train=False):
    """
    @param opt:
    @param is_train:
    @return:
    """
    # load pretrained model by default
    model = create_model(opt)
    device = model.device

    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"],
                     T=opt["sde"]["T"],
                     schedule=opt["sde"]["schedule"],
                     eps=opt["sde"]["eps"],
                     device=device)
    sde.set_model(model.model)

    return sde, model


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


def split_src_f_list(src_dir,
                     dst_dir,
                     n_threads,
                     ext=".jpg",
                     down_scale=2):
    """
    @param src_dir:
    @param dst_dir:
    @param n_threads:
    @param ext:
    @param down_scale:
    @return:
    """
    src_dir = os.path.abspath(src_dir)
    src_f_paths = []
    find_files(src_dir, src_f_paths, ext)
    print("\n[Info]: find total {:d} files of [{:s}]\n"
          .format(len(src_f_paths), ext))

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

    generated_f_paths = []
    find_files(dst_LR_dir, generated_f_paths, ext)
    generated_f_names = [os.path.split(x)[-1] for x in generated_f_paths]

    ungenerated_f_paths = [x for x in src_f_paths
                           if os.path.split(x)[-1] not in generated_f_names]
    print("[Info]: remain {:d} files not are not generated yet"
          .format(len(ungenerated_f_paths)))

    n_files_per_thread = len(ungenerated_f_paths) // n_threads
    thread_f_paths = []
    for i in range(n_threads):
        if i != n_threads - 1:
            thread_i_f_paths = ungenerated_f_paths[i * n_files_per_thread:
                                                   (i + 1) * n_files_per_thread]
        else:
            thread_i_f_paths = ungenerated_f_paths[i * n_files_per_thread:]
        thread_f_paths.append(thread_i_f_paths)
    return thread_f_paths


def gen_HR_LR_pairs(sde,
                    model,
                    src_f_list,
                    dst_dir,
                    down_scale=2):
    """
    @param sde:
    @param model:
    @param src_f_list:
    @param dst_dir:
    @param down_scale:
    @return:
    """
    if len(src_f_list) == 0:
        return

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

    with tqdm(total=len(src_f_list)) as p_bar:
        for src_f_path in src_f_list:
            src_f_name = os.path.split(src_f_path)[-1]

            # ----- generate HR image
            dst_hr_path = os.path.abspath(dst_HR_dir + "/" + src_f_name)
            if not os.path.isfile(dst_hr_path):
                shutil.copy(src_f_path, dst_HR_dir)  # copy HR img
                print("\n--> {:s} [cp to] {:s}\n".
                      format(src_f_name, dst_HR_dir))

            # ----- generate LR image
            hr = cv2.imread(src_f_path, cv2.IMREAD_COLOR)
            h, w, c = hr.shape
            lr = cv2.resize(hr, (w // down_scale, h // down_scale), cv2.INTER_CUBIC)
            dst_lr_path = os.path.abspath(dst_LR_sub_dir + "/" + src_f_name)
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
                      .format(src_f_name, dst_LR_sub_dir))
            p_bar.update()


def generate_LR_HR_pairs(sde,
                         model,
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


def run_degradation(sde,
                    model,
                    src_img_dir,
                    dst_img_dir,
                    ext=".jpg"):
    """
    @param sde:
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
                print("[Warning]: invalid file type: {:s}"
                      .format(img_name.split(".")[-1]))
                continue
            print("--> {:s} saved".format(dst_save_path))


def find_free_gpu():
    """
    :return:
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > tmp.py')
    memory_left_gpu = [int(x.split()[2]) for x in open('tmp.py', 'r').readlines()]
    most_free_gpu_idx = np.argmax(memory_left_gpu)
    # print(str(most_free_gpu_idx))
    return int(most_free_gpu_idx)


def select_device(device='', apex=False, batch_size=None):
    """
    :param device:
    :param apex:
    :param batch_size:
    :return:
    """
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), \
            'CUDA unavailable, invalid device %s requested' % device  # check availability

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print("[Info]: using CPU")

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


if __name__ == "__main__":
    # test_text2img(args, model, sde)
    # run_degradation(model,
    #                 src_img_dir="/mnt/diske/ROIs",
    #                 dst_img_dir="/mnt/diske/lyw/Degradations",
    #                 ext=".jpg")

    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt",
                        type=str,
                        default="./options/test/degradation.yml",
                        help="Path to options YMAL file.")
    parser.add_argument("-mt_gpu_ids",
                        type=str,
                        default="0,5",
                        help="")
    parser.add_argument("-s",
                        "--src_img_dir",
                        type=str,
                        default="",
                        help="")

    args = parser.parse_args()
    args = edict(vars(args))  # vars()函数返回对象object的属性和属性值的字典对象。

    n_threads = len(args.mt_gpu_ids.split(","))

    # ----------

    thread_f_paths = split_src_f_list(src_dir="/mnt/diske/RandomSamples",
                                      dst_dir="/mnt/ssd/lyw/SISR_data",
                                      n_threads=n_threads,
                                      ext=".jpg",
                                      down_scale=2)


    def task(gpu_id,
             opt_path,
             src_f_paths,
             dst_dir,
             down_scale=2):
        """
        @param gpu_id:
        @param opt_path:
        @param src_f_paths:
        @param dst_dir:
        @param down_scale:
        @return:
        """
        opt = option.parse_yaml(opt_path)
        opt["gpu_ids"] = [gpu_id]
        opt["dist"] = False
        opt["train"] = False
        opt["is_train"] = False
        opt["path"]["strict_load"] = True

        # ----- Set up the device
        # device = str(find_free_gpu())
        device = str(gpu_id)
        print("[Info]: using GPU {:s}.".format(device))
        device = select_device(device)
        opt.device = device

        sde, model = get_model(opt, is_train=False)
        gen_HR_LR_pairs(sde, model, src_f_paths, dst_dir, down_scale)


    gpu_ids = [int(x) for x in args.mt_gpu_ids.split(",")]

    # threads = []
    # for thread_i, gpu_id in enumerate(gpu_ids):
    #     thread = threading.Thread(target=task,
    #                               args=(gpu_id,
    #                                     args.opt,
    #                                     thread_f_paths[thread_i],
    #                                     "/mnt/ssd/lyw/SISR_data",
    #                                     ".jpg",
    #                                     2))
    #     thread.start()
    #     threads.append(thread)
    # for thread in threads:
    #     thread.join()

    for process_i, gpu_id in enumerate(gpu_ids):
        process = multiprocessing.Process(target=task,
                                          args=(gpu_id,
                                                args.opt,
                                                thread_f_paths[process_i],
                                                "/mnt/ssd/lyw/SISR_data",
                                                2))
        process.start()
