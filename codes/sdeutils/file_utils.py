# encoding=utf-8

import logging
import shutil

import math
import os
import random
import sys
import time
from collections import OrderedDict
from datetime import datetime
from shutil import get_terminal_size

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader

from tqdm import tqdm

from LicensePlateGenerator.generate_multi_plate import MultiPlateGenerator
from LicensePlateGenerator.generate_special_plate import generate_one_plate


def OrderedYaml():
    """
    @return:
    """
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def get_timestamp():
    """
    @return:
    """
    return datetime.now().strftime("%y%m%d-%H%M%S")


def mkdir(path):
    """
    @param path:
    @return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    """
    @param paths:
    @return:
    """
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    """
    @param path:
    @return:
    """
    if os.path.exists(path):
        new_name = path + "_archived_" + get_timestamp()
        print("Path already exists. Rename it to [{:s}]".format(new_name))
        logger = logging.getLogger("base")
        logger.info("Path already exists. Rename it to [{:s}]".format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    """
    @param seed:
    @return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name,
                 root,
                 phase,
                 level=logging.INFO,
                 screen=False,
                 tofile=False):
    """
    set up logger
    @param logger_name:
    @param root:
    @param phase:
    @param level:
    @param screen:
    @param tofile:
    @return:
    """
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + "_{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


class ProgressBar(object):
    """A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    """

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = bar_width if bar_width <= max_bar_width else max_bar_width
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print(
                "terminal width is too small ({}), please consider widen the terminal for better "
                "progressbar visualization".format(terminal_width)
            )
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write(
                "[{}] 0/{}, elapsed: 0s, ETA:\n{}\n".format(
                    " " * self.bar_width, self.task_num, "Start..."
                )
            )
        else:
            sys.stdout.write("completed: 0, elapsed: 0s")
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg="In progress..."):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = ">" * mark_width + "-" * (self.bar_width - mark_width)
            sys.stdout.write("\033[2F")  # cursor up 2 lines
            sys.stdout.write(
                "\033[J"
            )  # clean the output (remove extra chars since last display)
            sys.stdout.write(
                "[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n".format(
                    bar_chars,
                    self.completed,
                    self.task_num,
                    fps,
                    int(elapsed + 0.5),
                    eta,
                    msg,
                )
            )
        else:
            sys.stdout.write(
                "completed: {}, elapsed: {}s, {:.1f} tasks/s".format(
                    self.completed, int(elapsed + 0.5), fps
                )
            )
        sys.stdout.flush()


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


def gen_HQs(img_path_list_f, HQ_dir):
    """
    @param img_path_list_f:
    @param HQ_dir:
    @return:
    """
    img_path_list_f = os.path.abspath(img_path_list_f)
    if not os.path.isfile(img_path_list_f):
        print("[Info]: invalid file path list txt file: {:s}"
              .format(img_path_list_f))
        exit(-1)

    HQ_dir = os.path.abspath(HQ_dir)
    if os.path.isdir(HQ_dir):
        shutil.rmtree(HQ_dir)
        os.makedirs(HQ_dir)
        print("[Info]: {:s} made".format(HQ_dir))

    cnt = 0
    with open(img_path_list_f, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            items = line.split(" ")
            if len(items) > 1:
                file_path = items[0]
            elif len(items) == 1:
                file_path = items
            if not os.path.isfile(file_path):
                continue

            file_name = os.path.split(file_path)[-1]
            fields = file_name.split("_")
            if len(fields) < 3:
                continue

            plate_number = fields[0]
            plate_color = fields[1]
            plate_layers = fields[2]
            if '~' in plate_number \
                    or len(plate_number) < 7 \
                    or plate_layers == "double":
                continue
            if plate_color not in ["blue", "green", "yellow", "white", "black"]:
                continue
            if plate_color == "green" and len(plate_number) != 8:
                continue
            elif plate_color == "blue" and len(plate_number) != 7:
                continue

            lb = "_".join(fields)
            if "fake" in lb:
                continue
            print("--> label: {:s}".format(lb))

            hq_path = HQ_dir + "/{:s}".format(lb)
            os.path.abspath(hq_path)
            if not os.path.isfile(hq_path):
                shutil.copyfile(file_path, hq_path)
                print("--> {:s} [cp to] {:s}".format(file_name, HQ_dir))
            else:
                print("--> {:s} already exist".format(file_name))

            cnt += 1
    print("--> total {:d} valid samples found".format(cnt))
    print("[Info]: samples saved @ {:s}".format(HQ_dir))


def gen_LQs(HQ_dir, LQ_dir):
    """
    @param HQ_dir:
    @param LQ_dir:
    @return:
    """
    HQ_dir = os.path.abspath(HQ_dir)
    if not os.path.isdir(HQ_dir):
        print("[Info]: invalid real(HQ) license plate image dir:"
              .format(HQ_dir))
        exit(-1)
    LQ_dir = os.path.abspath(LQ_dir)
    if not os.path.abspath(LQ_dir):
        os.makedirs(LQ_dir)
        print("[Info]: {:s} made".format(LQ_dir))

    img_paths = []
    find_files(HQ_dir, img_paths, ".jpg")
    generator = MultiPlateGenerator('../LicensePlateGenerator/plate_model',
                                    '../LicensePlateGenerator/font_model')

    for img_path in img_paths:
        img_name = os.path.split(img_path)[-1]
        fields = img_name[:-4].split("_")
        plate_number = fields[0]
        bg_color = fields[1]
        double = fields[2]

        if bg_color == "green":
            bg_color = "green_car"

        if bg_color == "greenBus":
            bg_color = "green_truck"

        if double == "double":
            double = 1
        else:
            double = 0

        lq_path = LQ_dir + "/{:s}".format(img_name)
        generate_one_plate(generator,
                           plate_number,
                           bg_color,
                           double,
                           lq_path)


def gen_LQHQ(img_path_list_f,
             HQ_dir,
             LQ_dir):
    """
    @param img_path_list_f:
    @param HQ_dir:
    @param LQ_dir:
    @return:
    """
    img_path_list_f = os.path.abspath(img_path_list_f)
    if not os.path.isfile(img_path_list_f):
        print("[Info]: invalid file path list txt file: {:s}"
              .format(img_path_list_f))
        exit(-1)

    HQ_dir = os.path.abspath(HQ_dir)
    if os.path.isdir(HQ_dir):
        shutil.rmtree(HQ_dir)
    os.makedirs(HQ_dir)
    print("[Info]: {:s} made".format(HQ_dir))

    LQ_dir = os.path.abspath(LQ_dir)
    if os.path.isdir(LQ_dir):
        shutil.rmtree(LQ_dir)
    os.makedirs(LQ_dir)
    print("[Info]: {:s} made".format(LQ_dir))

    parent_dir = os.path.abspath(os.path.join(LQ_dir, ".."))

    # ----------
    generator = MultiPlateGenerator('../LicensePlateGenerator/plate_model',
                                    '../LicensePlateGenerator/font_model')

    total_to_be_checked = 0
    with open(img_path_list_f, "r", encoding="utf-8") as f:
        total_to_be_checked = len(f.readlines())

    cnt = 0
    with tqdm(total=total_to_be_checked) as progress_bar:
        with open(img_path_list_f, "r", encoding="utf-8") as f:
            for line in f.readlines():
                progress_bar.update()

                line = line.strip()
                items = line.split(" ")
                if len(items) > 1:
                    img_path = items[0]
                elif len(items) == 1:
                    img_path = items
                if not os.path.isfile(img_path):
                    continue

                img_name = os.path.split(img_path)[-1]
                fields = img_name.split("_")
                if len(fields) < 3:
                    continue

                plate_number = fields[0]
                plate_color = fields[1]
                plate_layers = fields[2]
                if '~' in plate_number \
                        or len(plate_number) < 7 \
                        or plate_layers == "double":
                    continue
                if plate_color not in ["blue", "green", "yellow", "white", "black"]:
                    continue
                if plate_color == "green" and len(plate_number) != 8:
                    continue
                elif plate_color == "blue" and len(plate_number) != 7:
                    continue

                lb = "_".join(fields)
                if "fake" in lb:
                    continue
                # print("--> label: {:s}".format(lb))

                # ---------- Generate HQ img: real img
                hq_path = HQ_dir + "/{:s}".format(lb)
                os.path.abspath(hq_path)
                if not os.path.isfile(hq_path):
                    hq_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    h, w, c = hq_img.shape
                    if w != 192 or h != 64:
                        hq_img = cv2.resize(hq_img, (192, 64), cv2.INTER_LINEAR)
                        cv2.imwrite(hq_path, hq_img)
                    else:
                        shutil.copyfile(img_path, hq_path)
                #     print("--> {:s} [cp to] {:s}".format(img_name, HQ_dir))
                # else:
                #     print("--> {:s} already exist".format(img_name))
                # print("--> HQ img {:s} generated @ {:s}"
                #       .format(img_name, HQ_dir))

                # ---------- Generate LQ img: ideal img
                if plate_color == "green":
                    plate_color = "green_car"

                if plate_color == "greenBus":
                    plate_color = "green_truck"

                if plate_layers == "double":
                    plate_layers = 1
                else:  # single
                    plate_layers = 0

                lq_path = LQ_dir + "/{:s}".format(img_name)
                lq_img = generator.generate_plate_special(plate_number, plate_color, plate_layers)
                if plate_layers == 0:  # resize
                    lq_img = cv2.resize(lq_img, (192, 64), cv2.INTER_LINEAR)
                cv2.imwrite(lq_path, lq_img)
                # print("--> LQ img {:s} generated @ {:s}\n"
                #       .format(img_name, LQ_dir))

                print("--> LQ-HQ pair {:s} generated @ {:s}".format(img_name, parent_dir))
                cnt += 1

                # ----------
    print("--> total {:d} valid sample pairs generated".format(cnt))


def augment_HQLQ_dataset(src_root, dst_root):
    """
    @param src_root: 
    @param dst_root: 
    @return: 
    """
    src_root = os.path.abspath(src_root)
    if not os.path.isdir(src_root):
        print("[Err]: invalid src root: {:s}".format(src_root))
        exit(-1)

    dst_root = os.path.abspath(dst_root)
    if not os.path.isdir(dst_root):
        print("[Err]: invalid dst root: {:s}".format(dst_root))
        exit(-1)

    src_img_paths = []
    find_files(src_root, src_img_paths, ".jpg")
    print("[Info]: total {:d} img files found".format(len(src_img_paths)))

    # ---------- check dst root dir
    dst_hq_dir = dst_root + "/HQ"
    dst_lq_dir = dst_root + "/LQ"

    dst_hq_img_paths = []
    find_files(dst_hq_dir, dst_hq_img_paths, ".jpg")
    dst_hq_img_names = [os.path.split(x)[-1] for x in dst_hq_img_paths]
    cnt = len(dst_hq_img_names)

    # ---------- define chinese license plate generator
    generator = MultiPlateGenerator('../LicensePlateGenerator/plate_model',
                                    '../LicensePlateGenerator/font_model')

    for img_path in src_img_paths:
        img_name = os.path.split(img_path)[-1]
        fields = img_name.split("_")
        new_name = img_name[:]  # copy

        # ---------- Generate LQ img: ideal img
        plate_number = fields[0]
        plate_color = fields[1]
        plate_layers = fields[2]
        if '~' in plate_number \
                or len(plate_number) < 7 \
                or plate_layers == "double":
            continue
        if plate_color not in ["blue", "green", "yellow", "white", "black"]:
            continue
        if plate_color == "green" and len(plate_number) != 8:
            continue
        elif plate_color == "blue" and len(plate_number) != 7:
            continue

        lb = "_".join(fields)
        if "fake" in lb:
            print("[Warning]: found fake HQ img!")
            # continue
        # print("--> label: {:s}".format(lb))

        if plate_color == "green":
            plate_color = "green_car"

        if plate_color == "greenBus":
            plate_color = "green_truck"

        if plate_layers == "double":
            plate_layers = 1
        else:  # single
            plate_layers = 0

        lq_path = dst_lq_dir + "/{:s}".format(new_name)
        lq_img = generator.generate_plate_special(plate_number, plate_color, plate_layers)
        if plate_layers == 0:  # resize
            lq_img = cv2.resize(lq_img, (192, 64), cv2.INTER_LINEAR)
        try:
            cv2.imwrite(lq_path, lq_img)
        except Exception as e:
            print(e)
            print("[Err]: lq_path: {:s}".format(lq_path))
            continue
        print("--> LQ img {:s} generated @ {:s}\n"
              .format(new_name, dst_lq_dir))

        # ----- Generate HQ img: real img
        if new_name not in dst_hq_img_names:  # direct copy
            shutil.copy(img_path, dst_hq_dir)
            print("--> {:s} [cp to] {:s}".format(new_name, dst_hq_dir))
        else:
            if len(fields) > 3:  # need new name
                new_id = int(fields[3][:-4]) + 1
                fields[3] = str(new_id)
                new_name = "_".join(fields)
            else:
                new_name = img_name[:-4] + "_1.jpg"
            dst_path = dst_hq_dir + "/" + new_name
            shutil.copyfile(img_path, dst_path)
            print("--> {:s} rename to {:s} copy to {:s}"
                  .format(img_name, new_name, dst_hq_dir))

        print("--> LQ-HQ pair {:s} generated @ {:s}\n".format(new_name, dst_root))
        cnt += 1


def filter_HQLQ_pairs(root_dir):
    """
    @param root_dir:
    @return:
    """
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        print("[Err]: invalid root dir: {:s}".format(root_dir))
        exit(-1)

    HQ_dir = os.path.abspath(root_dir + "/HQ")
    LQ_dir = os.path.abspath(root_dir + "/LQ")
    if not os.path.isdir(HQ_dir):
        print("[Err]: {:s} not exist".format(HQ_dir))
        exit(-1)
    if not os.path.isdir(LQ_dir):
        print("[Err]: {:s} not exist".format(LQ_dir))
        exit(-1)

    hq_file_paths, lq_file_paths = [], []
    find_files(HQ_dir, hq_file_paths, ".jpg")
    find_files(LQ_dir, lq_file_paths, ".jpg")
    hq_file_names = [os.path.split(x)[-1] for x in hq_file_paths]
    lq_file_names = [os.path.split(x)[-1] for x in lq_file_paths]

    if len(hq_file_paths) > len(lq_file_paths):
        for hq_name, hq_path in zip(hq_file_names, hq_file_paths):
            if hq_name in lq_file_names:
                continue
            else:
                if os.path.isfile(hq_path):
                    os.remove(hq_path)
                    print("--> {:s} removed".format(hq_path))

    elif len(hq_file_paths) < len(lq_file_paths):
        pass
    else:
        print("[Info]: equal HQ and LQ")


def gen_val_set(data_root, ratio=0.005):
    """
    @param data_root:
    @param ratio:
    @return:
    """
    data_root = os.path.abspath(data_root)
    if not os.path.isdir(data_root):
        os.makedirs(data_root)
        print("[Info]: {:s} invalid!")
        exit(-1)

    val_root = data_root + "/val"
    val_hq_dir = os.path.abspath(val_root + "/HQ")
    val_lq_dir = os.path.abspath(val_root + "/LQ")
    if os.path.isdir(val_hq_dir):
        shutil.rmtree(val_hq_dir)
    os.makedirs(val_hq_dir)
    print("[Info]: {:s} made".format(val_hq_dir))

    if os.path.isdir(val_lq_dir):
        shutil.rmtree(val_lq_dir)
    os.makedirs(val_lq_dir)
    print("[Info]: {:s} made".format(val_lq_dir))

    train_hq_dir = data_root + "/HQ"
    train_lq_dir = data_root + "/LQ"
    train_hq_paths = []
    find_files(train_hq_dir, train_hq_paths, ext=".jpg")
    print("[Info]: found {:d} img files".format(len(train_hq_paths)))
    train_lq_paths = [x.replace("HQ", "LQ") for x in train_hq_paths]
    cnt = 0
    for hq_path, lq_path in zip(train_hq_paths, train_lq_paths):
        if not (os.path.isfile(hq_path) and os.path.isfile(lq_path)):
            continue

        if np.random.random() <= ratio:
            hq_name = os.path.split(hq_path)[-1]
            lq_name = os.path.split(lq_path)[-1]

            val_hq_path = val_hq_dir + "/" + hq_name
            val_lq_path = val_lq_dir + "/" + lq_name
            shutil.copy(hq_path, val_hq_dir)
            print("--> {:s} [cp to] {:s}".format(hq_name, val_hq_dir))
            shutil.copy(lq_path, val_lq_dir)
            print("--> {:s} [cp to] {:s}".format(lq_name, val_lq_dir))
            cnt += 1
    print("[Info]: total {:d} samples".format(cnt))


def gen_lost_LQs(root_dir, ext=".jpg"):
    """
    @param root_dir:
    @param ext:
    @return:
    """
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        print("[Err]: invalid root dir: {:s}".format(root_dir))
        exit(-1)

    HQ_dir = os.path.abspath(root_dir + "/HQ")
    LQ_dir = os.path.abspath(root_dir + "/LQ")
    if not os.path.isdir(HQ_dir):
        print("[Err]: invalid HQ dir: {:s}".format(HQ_dir))
        exit(-1)
    if not os.path.isdir(LQ_dir):
        print("[Err]: invalid LQ dir: {:s}".format(LQ_dir))
        exit(-1)

    tmp_dir = os.path.abspath(root_dir + "/tmp")
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)
        print("[Info]: {:s} made".format(tmp_dir))

    hq_img_paths = []
    lq_img_paths = []
    find_files(HQ_dir, hq_img_paths, ext)
    find_files(LQ_dir, lq_img_paths, ext)
    hq_img_names = [os.path.split(x)[-1] for x in hq_img_paths]
    lq_img_names = [os.path.split(x)[-1] for x in lq_img_paths]

    generator = MultiPlateGenerator('../LicensePlateGenerator/plate_model',
                                    '../LicensePlateGenerator/font_model')

    with tqdm(total=len(hq_img_names)) as progress_bar:
        if len(hq_img_names) > len(lq_img_names):
            for hq_name in hq_img_names:
                if hq_name in lq_img_names:
                    progress_bar.update()
                    continue

                fields = hq_name.split("_")
                # ---------- Generate LQ img: ideal img
                plate_number = fields[0]
                plate_color = fields[1]
                plate_layers = fields[2]

                if '~' in plate_number \
                        or len(plate_number) < 7 \
                        or plate_layers == "double":
                    progress_bar.update()
                    print("[Warning]: ~ found in {:s}!".format(hq_name))
                    continue

                # if plate_color not in ["blue", "green", "yellow", "white", "black"]:
                #     progress_bar.update()
                #     continue
                if plate_color == "green" and len(plate_number) != 8:
                    progress_bar.update()
                    continue
                elif plate_color == "blue" and len(plate_number) != 7:
                    progress_bar.update()
                    continue

                lb = "_".join(fields)
                if "fake" in lb:
                    print("[Warning]: Fake hq found!")
                    # continue
                # print("--> label: {:s}".format(lb))

                if plate_color == "green":
                    plate_color = "green_car"

                if plate_color == "greenBus":
                    plate_color = "green_truck"

                if plate_layers == "double":
                    plate_layers = 1
                else:  # single
                    plate_layers = 0

                lq_path = LQ_dir + "/{:s}".format(hq_name)
                lq_img = generator.generate_plate_special(plate_number, plate_color, plate_layers)
                if plate_layers == 0:  # resize
                    lq_img = cv2.resize(lq_img, (192, 64), cv2.INTER_LINEAR)
                try:
                    cv2.imwrite(lq_path, lq_img)
                except Exception as e:
                    print("[Err]: lq_path: {:s}".format(lq_path))
                    progress_bar.update()
                    continue
                print("--> LQ img {:s} generated @ {:s}\n"
                      .format(hq_name, LQ_dir))
                progress_bar.update()


def viz_img_bursts(root_dir, out_dir, num_src_dirs=5, ext=".jpg"):
    """
    @param root_dir:
    @param out_dir:
    @return:
    """
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        print("[Err]: invalid root dir: {:s}".format(root_dir))
        exit(-1)

    out_dir = os.path.abspath(out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        print("[Info]: {:s} made".format(out_dir))

    img_paths = []
    src_img_names = []
    for i in range(num_src_dirs):
        src_i_dir = os.path.abspath(root_dir + "/Val_Dataset_{:d}".format(i + 1))
        if not os.path.isdir(src_i_dir):
            print("[Err]: invalid res dir path: {:s}".format(src_i_dir))
            exit(-1)

        src_i_dir_img_paths = [src_i_dir + "/" + x
                               for x in os.listdir(src_i_dir) if x.endswith(ext)
                               and "GEN" in x]
        src_i_dir_img_paths.sort()
        img_paths.append(src_i_dir_img_paths)

        if i == 0:
            src_img_names = [os.path.split(x)[-1] for x in src_i_dir_img_paths]

    img = cv2.imread(img_paths[0][0], cv2.IMREAD_COLOR)
    h, w, c = img.shape
    for j in range(len(src_img_names)):
        burst_img = np.zeros((h * num_src_dirs, w, c), img.dtype)  # H, W, C
        for i in range(num_src_dirs):
            img_path = img_paths[i][j]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            burst_img[i * h:(i + 1) * h, :, :] = img

        img_name = src_img_names[j]
        save_burst_path = out_dir + "/" + img_name
        cv2.imwrite(save_burst_path, burst_img)
        print("--> {:s} saved".format(save_burst_path))


def unify_HQ_LQ_imgsize(root_dir, img_size=(192, 64), ext=".jpg"):
    """
    @param root_dir:
    @param img_size:
    @return:
    """
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        print("[Err]: invalid root dir: {:s}".format(root_dir))
        exit(-1)

    HQ_dir = os.path.abspath(root_dir + "/HQ")
    LQ_dir = os.path.abspath(root_dir + "/LQ")
    if not os.path.isdir(HQ_dir):
        print("[Err]: invalid HQ dir: {:s}".format(HQ_dir))
        exit(-1)
    if not os.path.isdir(LQ_dir):
        print("[Err]: invalid LQ dir: {:s}".format(LQ_dir))
        exit(-1)

    tmp_dir = os.path.abspath(root_dir + "/tmp")
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)
        print("[Info]: {:s} made".format(tmp_dir))

    hq_img_paths = []
    lq_img_paths = []
    find_files(HQ_dir, hq_img_paths, ext)
    find_files(LQ_dir, lq_img_paths, ext)
    with tqdm(total=len(hq_img_paths)) as p_bar:
        for hq_path in hq_img_paths:
            img = cv2.imread(hq_path, cv2.IMREAD_COLOR)
            h, w, c = img.shape
            if w != img_size[0] or h != img_size[1]:
                img = cv2.resize(img, img_size, cv2.INTER_LINEAR)
                cv2.imwrite(hq_path, img)
                print("--> {:s} resized to {:d}×{:d}"
                      .format(hq_path, img_size[0], img_size[1]))
            p_bar.update()

    with tqdm(total=len(lq_img_paths)) as p_bar:
        for lq_path in lq_img_paths:
            img = cv2.imread(lq_path, cv2.IMREAD_COLOR)
            h, w, c = img.shape
            if w != img_size[0] or h != img_size[1]:
                img = cv2.resize(img, img_size, cv2.INTER_LINEAR)
                cv2.imwrite(lq_path, img)
                print("--> {:s} resized to {:d}×{:d}"
                      .format(lq_path, img_size[0], img_size[1]))
            p_bar.update()


def add_specify_plates_to_test(train_root, test_root, spe_str="使"):
    """
    @param train_root:
    @param test_root:
    @param spe_str:
    @return:
    """
    train_root = os.path.abspath(train_root)
    if not os.path.isdir(train_root):
        print("[Err]: invalid train root: {:s}".format(train_root))
        exit(-1)
    test_root = os.path.abspath(test_root)
    if not os.path.isdir(test_root):
        print("[Err]: invalid test root: {:s}".format(test_root))
        exit(-1)

    train_hq_dir = os.path.abspath(train_root + "/HQ")
    train_lq_dir = os.path.abspath(train_root + "/LQ")
    if not os.path.isdir(train_hq_dir):
        print("[Err]: invalid train HQ dir: {:s}".format(train_hq_dir))
        exit(-1)
    if not os.path.isdir(train_lq_dir):
        print("[Err]: invalid LQ dir: {:s}".format(train_lq_dir))
        exit(-1)

    test_hq_dir = os.path.abspath(test_root + "/HQ")
    test_lq_dir = os.path.abspath(test_root + "/LQ")
    if not os.path.isdir(test_hq_dir):
        print("[Err]: invalid test HQ dir: {:s}".format(test_hq_dir))
        exit(-1)
    if not os.path.isdir(test_lq_dir):
        print("[Err]: invalid test LQ dir: {:s}".format(test_lq_dir))
        exit(-1)

    train_hq_paths = []
    train_lq_paths = []
    find_files(train_hq_dir, train_hq_paths, ".jpg")
    find_files(train_lq_dir, train_lq_paths, ".jpg")
    train_hq_names = [os.path.split(x)[-1] for x in train_hq_paths]
    train_lq_names = [os.path.split(x)[-1] for x in train_lq_paths]

    assert len(train_hq_names) == len(train_lq_names)
    for hq_name, hq_path in zip(train_hq_names, train_hq_paths):
        if spe_str not in hq_name:
            continue
        if hq_name not in train_lq_names:
            print("[Err]: can not find {:s} in {:s}"
                  .format(hq_name, train_lq_dir))
            continue
        lq_path = hq_path.replace("HQ", "LQ")  # get src lq path
        if os.path.isfile(hq_path) and os.path.isfile(lq_path):
            dst_hq_path = test_hq_dir + "/" + hq_name
            dst_lq_path = test_lq_dir + "/" + hq_name
            if not os.path.isfile(dst_hq_path):
                shutil.copy(hq_path, test_hq_dir)
                print("--> {:s} [cp to] {:s}".format(hq_name, test_hq_dir))
            if not os.path.isfile(dst_lq_path):
                shutil.copy(lq_path, test_lq_dir)
                print("--> {:s} [cp to] {:s}".format(hq_name, test_lq_dir))


if __name__ == "__main__":
    # gen_HQs(img_path_list_f="../files/train_crnn_file_list221230.txt",
    #         HQ_dir="../../../HQ")

    # gen_LQHQ(img_path_list_f="../files/train_crnn_file_list221230.txt",
    #          HQ_dir="../../../img2img/HQ",
    #          LQ_dir="../../../img2img/LQ")
    #
    # filter_HQLQ_pairs(root_dir="../../../img2img/")
    #
    # augment_HQLQ_dataset(src_root="/mnt/diskd/even/LPDataSingle",
    #                      dst_root="../../../img2img")

    # gen_lost_LQs(root_dir="../../../img2img")

    # unify_HQ_LQ_imgsize(root_dir="../../../img2img")

    # gen_val_set(data_root="../../../img2img/",
    #             ratio=0.0020)

    viz_img_bursts(root_dir="../../results/img2img/ir-sde",
                   out_dir="/mnt/diske/vis_plate_gen_3",
                   num_src_dirs=5,
                   ext=".png")

    # add_specify_plates_to_test(train_root="../../../img2img",
    #                            test_root="../../../img2img/val")

    print("--> Done.")
