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
import time
import cv2
import random
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

import xml.etree.ElementTree as ET
from collections import Counter
from tqdm import tqdm

to_be_inserted_path = os.path.abspath("../../")
sys.path.insert(0, to_be_inserted_path)
print("[Info]: {:s} inserted".format(to_be_inserted_path))
from codes.LicensePlateGenerator.generate_multi_plate import MultiPlateGenerator
from codes.LicensePlateGenerator.generate_special_plate import generate_one_plate


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


def mkdir(dir_path):
    """
    @param dir_path:
    @return:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print("[Info]: {:s} made".format(dir_path))


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
    """
    A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    """

    def __init__(self, task_num=0, bar_width=50, start=True):
        """
        @param task_num:
        @param bar_width:
        @param start:
        """
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = bar_width if bar_width <= max_bar_width else max_bar_width
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        """
        @return:
        """
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print("terminal width is too small ({}), please consider widen the terminal for better "
                  "progressbar visualization"
                  .format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        """
        @return:
        """
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
        """
        @param msg:
        @return:
        """
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
            sys.stdout.write("[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n"
                             .format(bar_chars,
                                     self.completed,
                                     self.task_num,
                                     fps,
                                     int(elapsed + 0.5),
                                     eta,
                                     msg, ))
        else:
            sys.stdout.write(
                "completed: {}, elapsed: {}s, {:.1f} tasks/s"
                    .format(self.completed, int(elapsed + 0.5), fps))
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
                        or len(plate_number) < 7:
                    continue
                if plate_layers == "double":
                    print("\n[Warning]: double license plate not supported now!\n")
                    continue

                # if plate_color not in ["blue", "green", "yellow", "white", "black"]:
                #     continue
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
                if lq_img is None:
                    print("[Warning]: generate {:s} failed!"
                          .format(img_name))
                    continue
                if plate_layers == 0:  # resize
                    lq_img = cv2.resize(lq_img, (192, 64), cv2.INTER_LINEAR)
                cv2.imwrite(lq_path, lq_img)
                # print("--> LQ img {:s} generated @ {:s}\n"
                #       .format(img_name, LQ_dir))

                print("--> LQ-HQ pair {:s} generated @ {:s}"
                      .format(img_name, parent_dir))
                cnt += 1

                # ----------
    print("--> total {:d} valid sample pairs generated".format(cnt))


def augment_HQLQ_dataset(src, dst_root):
    """
    @param src: 
    @param dst_root: 
    @return: 
    """
    src = os.path.abspath(src)
    if os.path.isdir(src):
        print("[Info]: src dir: {:s}".format(src))
    elif os.path.isfile(src):
        print("[Info]: src file: {:s}".format(src))
    else:
        print("[Err]: invalid src: {:s}".format(src))
        exit(-1)

    dst_root = os.path.abspath(dst_root)
    if not os.path.isdir(dst_root):
        print("[Err]: invalid dst root: {:s}".format(dst_root))
        exit(-1)

    src_img_paths = []
    if os.path.isdir(src):
        find_files(src, src_img_paths, ".jpg")
    elif os.path.isfile(src):
        if src.endswith(".txt"):
            with open(src, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    img_path = line.strip()
                    if os.path.isfile(img_path):
                        src_img_paths.append(img_path)
        else:
            print("[Err]: invalid file: {:s}, it should be a txt file!"
                  .format(src))
            exit(-1)
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
    with tqdm(total=len(src_img_paths)) as p_bar:
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
                p_bar.update()
                continue
            # if plate_color not in ["blue", "green", "yellow", "white", "black"]:
            #     continue
            if plate_color == "green" and len(plate_number) != 8:
                p_bar.update()
                continue
            elif plate_color == "blue" and len(plate_number) != 7:
                p_bar.update()
                continue

            lb = "_".join(fields)
            if "fake" in lb:
                print("\n[Warning]: found fake HQ img!\n")
                p_bar.update()
                continue
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
            elif plate_layers == 1:
                print("\n[Warning]: double license plate not supported now!\n")
                p_bar.update()
                continue
            try:
                cv2.imwrite(lq_path, lq_img)
            except Exception as e:
                print(e)
                print("[Err]: lq_path: {:s}".format(lq_path))
                p_bar.update()
                continue
            print("\n--> LQ img {:s} generated @ {:s}\n"
                  .format(new_name, dst_lq_dir))

            # ----- Generate HQ img: real img
            if new_name not in dst_hq_img_names:  # direct copy
                shutil.copy(img_path, dst_hq_dir)
                print("\n--> {:s} [cp to] {:s}\n".format(new_name, dst_hq_dir))
            else:
                if len(fields) > 3:  # need new name
                    try:
                        new_id = int(fields[3][:-4]) + 1
                    except Exception as e:
                        print("[Exception]: {:s}".format(new_name))
                        print(e)
                        p_bar.update()
                        continue

                    fields[3] = str(new_id)
                    new_name = "_".join(fields)
                else:
                    new_name = img_name[:-4] + "_1.jpg"
                dst_path = dst_hq_dir + "/" + new_name
                shutil.copyfile(img_path, dst_path)
                print("\n--> {:s} rename to {:s} copy to {:s}\n"
                      .format(img_name, new_name, dst_hq_dir))

            print("--> LQ-HQ pair {:s} generated @ {:s}\n".format(new_name, dst_root))
            cnt += 1
            p_bar.update()


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
        with tqdm(total=len(hq_file_names)) as p_bar:
            for hq_name, hq_path in zip(hq_file_names, hq_file_paths):
                if hq_name in lq_file_names:
                    p_bar.update()
                    continue
                else:
                    if os.path.isfile(hq_path):
                        os.remove(hq_path)
                        print("\n--> {:s} removed\n".format(hq_path))
                        p_bar.update()

    elif len(hq_file_paths) < len(lq_file_paths):
        with tqdm(total=len(lq_file_names)) as p_bar:
            for lq_name, lq_path in zip(lq_file_names, lq_file_paths):
                if lq_name in hq_file_names:
                    p_bar.update()
                    continue
                else:
                    if os.path.isfile(lq_path):
                        os.remove(lq_path)
                        print("\n--> {:s} removed\n".format(lq_path))
                        p_bar.update()
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


def cvt_png_to_jpg(src_dir):
    """
    :param src_dir:
    :return:
    """
    src_dir = os.path.abspath(src_dir)
    if not os.path.isdir(src_dir):
        print("[Err]: invalid src dir: {:s}".format(src_dir))
        exit(-1)
    print("[Info]: src dir: {:s}".format(src_dir))

    png_file_paths = []
    find_files(src_dir, png_file_paths, ".png")
    print("[Info]: find {:d} png files".format(len(png_file_paths)))

    for png_f_path in png_file_paths:
        img = cv2.imdecode(np.fromfile(png_f_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        jpg_f_path = png_f_path.replace(".png", ".jpg")
        cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tofile(jpg_f_path)
        # cv2.imwrite("output2.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 无损
        print("--> {:s} written".format(jpg_f_path))


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
    cnt = 0
    with tqdm(total=len(hq_img_names)) as p_bar:
        if len(hq_img_names) > len(lq_img_names):
            for hq_name in hq_img_names:
                if hq_name in lq_img_names:
                    p_bar.update()
                    continue

                fields = hq_name.split("_")

                # ---------- Generate LQ img: ideal img
                plate_number = fields[0]
                plate_color = fields[1]
                plate_layers = fields[2]

                if '~' in plate_number:
                    p_bar.update()
                    continue

                if len(plate_number) < 7:
                    print("\n[Warning]: found number of chars < 7!\n")
                    p_bar.update()
                    continue

                if plate_layers == "double":
                    print("\n[Warning]: found [double] license plate {:s}!\n".format(hq_name))
                    p_bar.update()
                    continue

                # if plate_color not in ["blue", "green", "yellow", "white", "black"]:
                #     p_bar.update()
                #     continue
                if plate_color == "green" and len(plate_number) != 8:
                    p_bar.update()
                    continue
                elif plate_color == "blue" and len(plate_number) != 7:
                    p_bar.update()
                    continue

                lb = "_".join(fields)
                if "fake" in lb:
                    print("\n[Warning]: Fake hq found!\n")
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
                try:
                    lq_img = generator.generate_plate_special(plate_number,
                                                              plate_color,
                                                              plate_layers)
                except Exception as e:
                    print("\n", e)
                    print("\n[Warning]: generated {:s} failed!\n"
                          .format(hq_name))
                    continue
                if lq_img is None:
                    print("\n[Err]: {:s} generated failed!\n"
                          .format(hq_name))
                    p_bar.update()
                    continue
                if plate_layers == 0:  # resize
                    lq_img = cv2.resize(lq_img, (192, 64), cv2.INTER_LINEAR)
                elif plate_layers == 1:
                    print("\n[Warning]: double license plate not supported yet!\n")
                    p_bar.update()
                    continue
                try:
                    cv2.imwrite(lq_path, lq_img)
                except Exception as e:
                    print("\n[Err]: lq_path: {:s}\n".format(lq_path))
                    p_bar.update()
                    continue
                print("\n--> LQ img {:s} generated @ {:s}\n"
                      .format(hq_name, LQ_dir))

                cnt += 1
                p_bar.update()


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
            hq_name = os.path.split(hq_path)[-1]
            fields = hq_name.split("_")
            plate_number = fields[0]
            plate_color = fields[1]
            plate_layer = fields[2]
            if plate_number == "double" or plate_layer == "Double":
                os.remove(hq_path)
                print("\n[Warning]: {:s} removed\n".format(hq_path))
                p_bar.update()
                continue
            if "double" in hq_name or "Double" in hq_name:
                os.remove(hq_path)
                print("\n[Warning]: {:s} removed\n".format(hq_path))
                p_bar.update()
                continue
            img = cv2.imread(hq_path, cv2.IMREAD_COLOR)
            h, w, c = img.shape
            if w != img_size[0] or h != img_size[1]:
                img = cv2.resize(img, img_size, cv2.INTER_LINEAR)
                cv2.imwrite(hq_path, img)
                print("\n--> {:s} resized to {:d}×{:d}\n"
                      .format(hq_path, img_size[0], img_size[1]))
            p_bar.update()

    with tqdm(total=len(lq_img_paths)) as p_bar:
        for lq_path in lq_img_paths:
            lq_name = os.path.split(lq_path)[-1]
            fields = lq_name.split("_")
            plate_number = fields[0]
            plate_color = fields[1]
            plate_layer = fields[2]
            if plate_number == "double" or plate_layer == "Double":
                os.remove(lq_path)
                print("\n[Warning]: {:s} removed\n".format(lq_path))
                p_bar.update()
                continue
            if "double" in hq_name or "Double" in lq_name:
                os.remove(lq_path)
                print("\n[Warning]: {:s} removed\n".format(lq_path))
                p_bar.update()
                continue
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


def viz_txt2img_set(src_dir, viz_dir, ext=".png"):
    """
    @param src_dir:
    @param viz_dir:
    @param ext:
    @return:
    """
    src_dir = os.path.abspath(src_dir)
    if not os.path.isdir(src_dir):
        print("[Err]: invalid src dir: {:s}".format(src_dir))
        exit(-1)

    viz_dir = os.path.abspath(viz_dir)
    if os.path.isdir(viz_dir):
        shutil.rmtree(viz_dir)
    os.makedirs(viz_dir)
    print("[Info]: {:s} made".format(viz_dir))

    all_img_paths = []
    find_files(src_dir, all_img_paths, ext)

    img_set = set()
    for img_path in all_img_paths:
        if not os.path.isfile(img_path):
            print("[Warning]: {:s} not exist!")
            continue

        img_name = os.path.split(img_path)[-1]
        fields = img_name.split("_")
        assert len(fields) > 3
        img_set.add("_".join(fields[:-1]))
    print("[Info]: total {:d} img sets to be visualized"
          .format(len(img_set)))

    for unique_img_name in img_set:
        unique_img_paths = [x for x in all_img_paths if unique_img_name in x]
        img = cv2.imread(unique_img_paths[0], cv2.IMREAD_COLOR)
        h, w, c = img.shape

        img_burst = np.zeros((h * len(unique_img_paths), w, c), dtype=img.dtype)
        for img_i, img_path in enumerate(unique_img_paths):
            if img_i == 0:
                img_burst[:h, :, :] = img
            else:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img_burst[img_i * h:(img_i + 1) * h, :, :] = img
        viz_path = viz_dir + "/" + unique_img_name + ext
        cv2.imwrite(viz_path, img_burst)
        print("--> {:s} saved".format(viz_path))


global char_set, letter_set, province_set, special_set
char_set = {
    "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z",
    "桂", "贵", "冀", "吉", "京",
    "琼", "陕", "苏", "湘", "渝",
    "豫", "藏", "川", "鄂", "甘",
    "赣", "黑", "沪", "津", "晋",
    "鲁", "蒙", "闽", "宁", "青",
    "使", "皖", "新", "粤", "云",
    "浙", "辽", "军", "空", "兰",
    "广", "海", "成", "应", "急",
    "学", "警", "港", "澳", "赛",
    "领", "挂"
}

letter_set = {
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z",
}

province_set = {
    "桂", "贵", "冀", "吉", "京",
    "琼", "陕", "苏", "湘", "渝",
    "豫", "藏", "川", "鄂", "甘",
    "赣", "黑", "沪", "津", "晋",
    "鲁", "蒙", "闽", "宁", "青",
    "皖", "新", "粤", "云", "浙",
    "辽"
}

special_set = {
    "军", "空", "兰",
    "广", "海", "成", "应", "急",
    "学", "警", "港", "澳", "赛",
    "领", "挂", "WJ"
}


def split_and_statistics(root_dir):
    """
    @param root_dir:
    @return:
    """
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        print("[Info]: invalid root dir:{:s}".format(root_dir))
        exit(-1)

    LQ_dir = os.path.abspath(root_dir + "/LQ")
    HQ_dir = os.path.abspath(root_dir + "/HQ")
    if not os.path.abspath(LQ_dir):
        print("[Info]: invalid LQ dir: {:s}".format(LQ_dir))
        exit(-1)
    if not os.path.abspath(HQ_dir):
        print("[Info]: invalid HQ dir: {:s}".format(HQ_dir))
        exit(-1)
    # parent_dir = os.path.abspath(os.path.join(LQ_dir, ".."))

    tmp_dir = os.path.abspath(root_dir + "/tmp")
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)
        print("[Info]: {:s} made".format(tmp_dir))

    hq_img_paths = []
    find_files(HQ_dir, hq_img_paths, ".jpg")
    print("[Info]: finding {:d} files".format(len(hq_img_paths)))

    # ---------- 构建统计词典
    special_plate_dict = {
        "港": 0,
        "澳": 0,
        "大使馆": 0,
        "应急": 0,
        "警_武警": 0,
        "军": 0,
    }

    # 构建【特殊】车牌子目录
    for k in special_plate_dict.keys():
        sub_dir = os.path.abspath(root_dir + "/" + k)
        if not os.path.isdir(sub_dir):
            os.makedirs(sub_dir)
            print("[Info]: {:s} made".format(sub_dir))

    # 构建【省份】车牌字典
    province_plate_dict = dict.fromkeys(province_set, 0)

    # 构建【省份】车牌子目录
    for name in province_set:
        sub_dir = os.path.abspath(root_dir + "/" + name)
        if not os.path.isdir(sub_dir):
            os.makedirs(sub_dir)
            print("[Info]: {:s} made".format(sub_dir))

    cnt_invalid = 0
    with tqdm(total=len(hq_img_paths)) as p_bar:
        for img_path in hq_img_paths:
            img_name = os.path.split(img_path)[-1]
            fields = img_name.split("_")
            assert len(fields) >= 3

            plate_number = fields[0]
            plate_color = fields[1]
            plate_layer = fields[2]

            # ---------- 先检查是否属于特殊车牌
            if "港" in plate_number:
                special_plate_dict["港"] += 1
                dst_dir = root_dir + "/港"
            if "澳" in plate_number:
                special_plate_dict["澳"] += 1
                dst_dir = root_dir + "/澳"
            elif "使" in plate_number:
                special_plate_dict["大使馆"] += 1
                dst_dir = root_dir + "/大使馆"
            elif "应急" in plate_number:
                special_plate_dict["应急"] += 1
                dst_dir = root_dir + "/应急"
            elif "警" in plate_number or "WJ" in plate_number:
                special_plate_dict["警_武警"] += 1
                dst_dir = root_dir + "/警_武警"
            elif plate_color == "white" and plate_number[0] in letter_set:
                special_plate_dict["军"] += 1
                dst_dir = os.path.abspath(root_dir + "/军")
            else:  # ---------- 再对省份进行划分
                if plate_number[0] in province_set:  # 首字符判断
                    province = plate_number[0]
                    province_plate_dict[province] += 1
                    dst_dir = root_dir + "/{:s}".format(province)
                else:
                    print("\n[Warning]: invalid plate: {:s}!\n".format(img_name))

                    # ----- cp to 【root/tmp】 dir
                    dst_path = os.path.abspath(tmp_dir + "/" + img_name)
                    if not os.path.isfile(dst_path):
                        shutil.copyfile(img_path, dst_path)
                        print("\n--> {:s} [cp to] {:s}\n".format(img_name, tmp_dir))
                    cnt_invalid += 1
                    p_bar.update()
                    continue

            # ---------- Copy file
            # if os.path.isdir(dst_dir):
            #     print("[Err]: invalid dir: {:s}".format(dst_dir))
            #     p_bar.update()
            #     continue
            dst_path = os.path.abspath(dst_dir + "/" + img_name)
            if not os.path.isfile(dst_path):
                shutil.copy(img_path, dst_dir)
                print("\n--> {:s} [cp to] {:s}\n".format(img_name, dst_dir))

            p_bar.update()
            # ---------- 遍历结束

    # 统计信息
    cnt = 0
    for k, v in province_plate_dict.items():
        print("{:8s} {:6d}".format(k + ":", v))
        cnt += v
    for k, v in special_plate_dict.items():
        print("{:8s} {:6d}".format(k + ":", v))
        cnt += v
    print("[Info]: total {:d} valid files".format(cnt))


def rename_LPs(root_dir):
    """
    @param root_dir:
    @return:
    """
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        print("[Info]: invalid root dir:{:s}".format(root_dir))
        exit(-1)

    LQ_dir = os.path.abspath(root_dir + "/LQ")
    HQ_dir = os.path.abspath(root_dir + "/HQ")
    if not os.path.abspath(LQ_dir):
        print("[Info]: invalid LQ dir: {:s}".format(LQ_dir))
        exit(-1)
    if not os.path.abspath(HQ_dir):
        print("[Info]: invalid HQ dir: {:s}".format(HQ_dir))
        exit(-1)

    hq_img_paths = []
    find_files(HQ_dir, hq_img_paths, ".jpg")
    print("[Info]: find {:d} files in {:s}"
          .format(len(hq_img_paths), HQ_dir))

    with tqdm(total=len(hq_img_paths)) as p_bar:
        for img_path in hq_img_paths:
            img_name = os.path.split(img_path)[-1]
            fields = img_name.split("_")
            plate_number = fields[0]
            plate_color = fields[1]
            plate_layer = fields[2]

            if plate_color == "whitearmy":
                plate_color = "white"

            if plate_number[0] in letter_set \
                    and plate_number[1] in letter_set \
                    and len(plate_number) == 7 \
                    and plate_color != "white":
                plate_color = "white"

            if plate_layer == "False" \
                    or plate_layer == "false":
                plate_layer = "single"

            fields[0] = plate_number
            fields[1] = plate_color
            fields[2] = plate_layer

            new_name = "_".join(fields)
            new_path = HQ_dir + "/" + new_name
            if img_path == new_path:
                p_bar.update()
                continue
            os.rename(img_path, new_path)
            print("\n--> {:s} [renamed to] {:s} in {:s}\n"
                  .format(img_name, new_name, HQ_dir))
            p_bar.update()


def rename_no_suffix_files(dir_path, ext=".jpg"):
    """
    @param dir_path:
    @param ext:
    @return:
    """
    dir_path = os.path.abspath(dir_path)
    if not os.path.isdir(dir_path):
        print("[Err]: invalid dir: {:s}".format(dir_path))
        exit(-1)

    no_suffix_file_paths = [x for x in os.listdir(dir_path)
                            if os.path.splitext(x)[-1] == ""]
    print("[Info]: find {:d} files without suffix".format(len(no_suffix_file_paths)))

    with tqdm(total=len(no_suffix_file_paths)) as p_bar:
        for old_name in no_suffix_file_paths:
            old_path = dir_path + "/" + old_name
            if not os.path.isfile(old_path):
                print("\n[Warning]: invalid old file path: {:s}\n"
                      .format(old_path))
                p_bar.update()
                continue
            fields = old_name.split("_")
            plate_number = fields[0]
            plate_color = fields[1]
            plate_layer = fields[2]
            old_base_name = "_".join([plate_number, plate_color, plate_layer])
            time_str = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
            time.sleep(1)
            new_name = old_base_name + "_" + time_str + ext
            new_path = dir_path + "/" + new_name
            os.rename(old_path, new_path)
            print("\n--> {:s} [renamed to] {:s} in {:s}\n"
                  .format(old_name, new_name, dir_path))
            p_bar.update()


global plate_color_mapping
plate_color_mapping = {
    "蓝": "blue",
    "白": "white",
    "黑": "black",
    "绿": "green",
    "黄": "yellow",
}


def parseXmlForPlate(img_path,
                     xml_path,
                     dst_dir,
                     counter,
                     ext=".jpg",
                     clock_wise=True,
                     split_dir=False,
                     logging=True):
    """
    Args:
        img_path:
        xml_path:
        dst_dir:
        counter:
        ext,
        clock_wise:
        split_dir:
        logging:
    Returns:
    """
    assert os.path.isdir(dst_dir)

    img_path = os.path.abspath(img_path)
    if not os.path.isfile(img_path):
        print("[Err]: invalid jpg path: {:s}.".format(os.path.abspath(img_path)))
        return -1
    xml_path = os.path.abspath(xml_path)
    if not os.path.isfile(xml_path):
        print("\n[Warning]: invalid xml path {:s}.\n".format(os.path.abspath(xml_path)))
        return -1

    ## --------- Read in image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print("[Err]: read img {:s} failed!"
              .format(img_path))
        return -1

    ## ---------- parse xml
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_info = f.read()

        try:
            root = ET.fromstring(xml_info)
        except Exception as e:
            print("[Err]: can not parse xml file: {:s}".format(xml_path))

        # obj = root.find('markNode')
        # if obj is None:
        #     print("[Err]: find markNode failed!")
        #     return

        for obj in root.iter('object'):  # 遍历object, 每个object代表一个标注目标
            target_type = obj.find('targettype')
            target_name = target_type.text
            if target_name != "plate":  # 读取车牌对象
                continue

            ## ----- 解析车牌属性
            color_node = obj.find("color")
            if color_node is None:
                print("[Warning]: can not find color!")
                continue
            plate_color = color_node.text
            plate_color = plate_color_mapping[plate_color]

            type_node = obj.find("mode")
            if type_node is None:
                print("[Warning]: can not find type!")
                continue
            plate_type = type_node.text

            ## ----- 解析车牌号码
            char_nodes = obj.find("characters")
            if char_nodes is None:
                print("[Warning]: can not find character!")
                continue

            plate_number = []
            for char in char_nodes.iter("char"):
                char = char.find("data").text
                plate_number.append(char)
            n_chars = len(plate_number)
            plate_number = "".join(plate_number)

            ## ----- 解析4个角点: 左上角逆时针
            vertex_nodes = obj.find("vertexs")
            if vertex_nodes is None:
                print("[Warning]: can not find vertex!")
                continue

            vertices = []
            for vertex in vertex_nodes.iter("vertex"):
                x = int(vertex.find("x").text)
                y = int(vertex.find("y").text)
                vertices.append([x, y])
            assert len(vertices) == 4

            # draw4Corners(img, vertices)
            # cv2.imwrite(vis_path, img)
            # print("{:s} written.".format(vis_path))

            # 基于4个角点Warp到正视图
            if len(vertices) != 4:
                print("[Warning]: corners number is not 4!.")
                continue

            vertex_arr = np.array(vertices)
            pt_ord = [min(vertex_arr[:, 0]), min(vertex_arr[:, 1])]
            pt_ord[0] = pt_ord[0] if pt_ord[0] >= 0 else 0
            pt_ord[1] = pt_ord[1] if pt_ord[1] >= 0 else 0

            rect_w = max(vertex_arr[:, 0]) - min(vertex_arr[:, 0])
            rect_h = max(vertex_arr[:, 1]) - min(vertex_arr[:, 1])
            if rect_w <= 0 or rect_h <= 0:
                print("[Warning]: invalid Rect, rect_w: {:d}, rect_h: {:d}"
                      .format(rect_w, rect_h))
                continue

            rect = img[pt_ord[1]: pt_ord[1] + rect_h, pt_ord[0]: pt_ord[0] + rect_w, :]
            h, w = rect.shape[:2]
            if h == 0 or w == 0:
                print("[Warning]: invalid Rect of plate!")
                continue

            src_pts, dst_pts = [], []

            if plate_type == "单":
                for i, vertex in enumerate(vertex_arr):
                    src_pts.append([vertex[0] - pt_ord[0], vertex[1] - pt_ord[1]])  # 转换到车牌局部坐标系

                    if clock_wise:  # 顺时针标注的4角点
                        if i == 0:
                            dst_pts.append([0, 0])
                        elif i == 1:
                            dst_pts.append([191, 0])
                        elif i == 2:
                            dst_pts.append([191, 63])
                        elif i == 3:
                            dst_pts.append([0, 63])

                    else:  # 逆时针标注的4角点
                        if i == 0:
                            dst_pts.append([0, 0])
                        elif i == 1:
                            dst_pts.append([0, 63])
                        elif i == 2:
                            dst_pts.append([191, 63])
                        elif i == 3:
                            dst_pts.append([191, 0])

            elif plate_type == "双":
                for i, vertex in enumerate(vertex_arr):
                    src_pts.append([vertex[0] - pt_ord[0], vertex[1] - pt_ord[1]])  # 转换到车牌局部坐标系

                    if clock_wise:
                        if i == 0:
                            dst_pts.append([0, 0])
                        elif i == 1:
                            dst_pts.append([191, 0])
                        elif i == 2:
                            dst_pts.append([191, 95])
                        elif i == 3:
                            dst_pts.append([0, 95])

                    else:
                        if i == 0:
                            dst_pts.append([0, 0])
                        elif i == 1:
                            dst_pts.append([0, 95])
                        elif i == 2:
                            dst_pts.append([191, 95])
                        elif i == 3:
                            dst_pts.append([191, 0])

            if not (len(src_pts) == 4 and len(dst_pts)) == 4:
                print("[Warning]: invalid src_pts or dst_pts!")
                continue

            src_pts, dst_pts = np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32)
            mat = cv2.getPerspectiveTransform(src_pts, dst_pts)

            if plate_type == "单":
                try:
                    plate_img_warped = cv2.warpPerspective(rect, mat, (192, 64))
                except Exception as e:
                    print(e)
                    continue
            elif plate_type == "双":
                try:
                    plate_img_warped = cv2.warpPerspective(rect, mat, (192, 96))
                except Exception as e:
                    print(e)
                    continue

            ## ----- save warped image
            if plate_type == "单":
                plate_type = "single"
            elif plate_type == "双":
                plate_type = "double"

            if split_dir:
                save_dir = dst_dir + "/" + plate_number
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
            else:
                save_dir = dst_dir

            pl_number = plate_number.replace("*", "_")  # .replace("~", ".")
            time_str = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
            time.sleep(0.1)
            # save_path = save_dir + "/" \
            #             + pl_number \
            #             + "_" + plate_color \
            #             + "_" + plate_type \
            #             + "_" + str(counter["id"]) + ext
            save_path = save_dir + "/" \
                        + pl_number \
                        + "_" + plate_color \
                        + "_" + plate_type \
                        + "_" + time_str + ext
            if not os.path.isfile(save_path):
                # cv2.imwrite(save_path, plate_img_warped)
                if ext == ".jpg":
                    cv2.imencode(".jpg", plate_img_warped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tofile(save_path)
                elif ext == ".png":
                    # cv2.imencode(".png", plate_img_warped, [cv2.IMWRITE_PNG_COMPRESSION, 0])[1].tofile(save_path)
                    save_path = save_path.replace(".png", ".jpg")
                    cv2.imencode(".jpg", plate_img_warped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tofile(save_path)
                else:
                    cv2.imencode(ext, plate_img_warped)[1].tofile(save_path)

                print("\n--> {:s} saved\n".format(save_path))

            if logging:
                print("\n")
                print("Rect width: ", rect_w)
                print("Rect height: ", rect_h)
                print("车牌类型: ", plate_type)
                print("车牌颜色: ", plate_color)
                print("车牌号码: {:s}, {:d}字符".format(plate_number, n_chars))
                print("\n")

            ## ----- sample counter updating
            counter.update(["id"])

        return 0


def parsePlates(src_dir,
                dst_dir,
                clock_wise=False,
                ext=".jpg",
                split_dir=False,
                logging=True):
    """
    Args:
        src_dir:
        dst_dir:
        clock_wise: True: clock_wise | False: anti-clock_wise
        ext:
        split_dir:
    Returns:
    """
    src_dir = os.path.abspath(src_dir)
    if not os.path.isdir(src_dir):
        print("[Err]: invalid src dir.")
        return

    dst_dir = os.path.abspath(dst_dir)
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)
    print("{:s} made.".format(dst_dir))

    img_list = []
    find_files(src_dir, img_list, ext=ext)
    img_list.sort()
    xml_list = [x.replace(ext, ".xml") for x in img_list]

    item_list = [(img_path, xml_path) for img_path, xml_path in zip(img_list, xml_list)]
    print("[Info]: Total {:d} valid items found.".format(len(item_list)))

    counter = Counter({"id": 0})
    with tqdm(total=len(item_list)) as p_bar:
        for i, (img_path, xml_path) in enumerate(item_list):
            ## ----- Parsing & Warping
            ret = parseXmlForPlate(img_path,
                                   xml_path,
                                   dst_dir,
                                   counter,
                                   ext,
                                   clock_wise,
                                   split_dir,
                                   logging)
            # if ret < 0:
            #     p_bar.update()
            #     continue
            ## -----

            p_bar.update()


def run_parsing_plates(src_root_dir,
                       dst_root_dir,
                       clock_wise=False):
    """
    @param src_root_dir:
    @param dst_root_dir:
    @param clock_wise:
    @return:
    """
    src_root_dir = os.path.abspath(src_root_dir)
    if not os.path.abspath(src_root_dir):
        print("[Err]: invalid src root dir: {:s}"
              .format(src_root_dir))
        exit(-1)
    dst_root_dir = os.path.abspath(dst_root_dir)
    if not os.path.abspath(dst_root_dir):
        try:
            os.makedirs(dst_root_dir)
        except Exception as e:
            print(e)
            exit(-1)

    sub_src_dirs = [src_root_dir + "/" + x for x in os.listdir(src_root_dir)]
    for sub_src_dir in sub_src_dirs:
        dir_name = os.path.split(sub_src_dir)[-1]
        sub_dst_dir = dst_root_dir + "/" + dir_name
        print("--> processing {:s}...".format(sub_src_dir))
        parsePlates(src_dir=sub_src_dir,
                    dst_dir=sub_dst_dir,
                    clock_wise=clock_wise)


def cp_files(src_dir, dst_dir, ext=".jpg"):
    """
    @param src_dir:
    @param dst_dir:
    @param ext:
    @return:
    """
    src_dir = os.path.abspath(src_dir)
    if not os.path.abspath(src_dir):
        print("[Err]: invalid src dir: {:s}".format(src_dir))
        exit(-1)

    dst_dir = os.path.abspath(dst_dir)
    if not os.path.abspath(dst_dir):
        print("[Err]: invalid dst dir: {:s}".format(dst_dir))
        exit(-1)

    src_file_paths = []
    find_files(src_dir, src_file_paths, ext)
    print("[Info]: find {:d} {:s} files"
          .format(len(src_file_paths), ext))

    for src_f_path in src_file_paths:
        if not os.path.isfile(src_f_path):
            print("[Warning]: invalid file path: {:s}"
                  .format(src_f_path))
            continue
        src_f_name = os.path.split(src_f_path)[-1]

        dst_f_path = dst_dir + "/" + src_f_name
        if not os.path.isfile(dst_f_path):
            shutil.copy(src_f_path, dst_dir)
            print("--> {:s} [cp to] {:s}"
                  .format(src_f_name, dst_dir))


def split_to_be_trained_set(file_list_path,
                            dst_dir,
                            n_files_per_sub_dir=300):
    """
    @param file_list_path:
    @param dst_dir:
    @param n_files_per_sub_dir:
    @return:
    """
    file_list_path = os.path.abspath(file_list_path)
    if not os.path.abspath(file_list_path):
        print("[Err]: invalid file list path: {:s}"
              .format(file_list_path))
        exit(-1)

    dst_dir = os.path.abspath(dst_dir)
    if not os.path.isdir(dst_dir):
        print("[Warning]: invalid dst dir: {:s}".format(dst_dir))
        os.makedirs(dst_dir)
        print("[Info]: {:s} made".format(dst_dir))

    file_paths_list = []
    with open(file_list_path, "r", encoding="utf-8") as f:

        for line in f.readlines():
            f_path = line.strip()
            file_paths_list.append(f_path)

        n_files = len(file_paths_list)
        print("[Info]: total {:d} files to be processed..."
              .format(n_files))
        n_sub_dirs = n_files // n_files_per_sub_dir + 1

    random.shuffle(file_paths_list)
    with tqdm(total=n_files) as p_bar:
        for i, f_path in enumerate(file_paths_list):
            if not os.path.isfile(f_path):
                print("[Warning]: {:s} invalid".format(f_path))
                p_bar.update()
                continue

            j_th_sub_dir = i // n_files_per_sub_dir
            sub_dir_path = os.path.abspath(dst_dir + "/{:05d}".format(j_th_sub_dir))
            if not os.path.isdir(sub_dir_path):
                os.makedirs(sub_dir_path)
                print("\n[Info]: {:s} made\n".format(sub_dir_path))

            time_str = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
            f_base_name = os.path.split(f_path)[-1]
            fields = f_base_name.split("_")
            plate_number = fields[0]
            plate_color = fields[1]
            if "." in fields[2]:
                plate_layer, ext = fields[2].split(".")
            else:
                plate_layer = fields[2]
                ext = f_base_name.split(".")[-1]
            new_f_name = plate_number + "_" + plate_color + "_" + plate_layer \
                         + "_" + time_str + ".{:s}".format(ext)
            new_f_path = sub_dir_path + "/{:s}".format(new_f_name)
            if not os.path.isfile(new_f_path):
                shutil.copyfile(f_path, new_f_path)
                # print("\n--> {:s} [cp to] {:s}\n".format(f_base_name, sub_dir_path))
            time.sleep(0.05)
            p_bar.update()


def random_sample_imgs(src_dir,
                       ext=".jpg",
                       ratio=0.001):
    """
    @param src_dir:
    @param ext:
    @param ratio:
    @return:
    """
    src_dir = os.path.abspath(src_dir)
    if not os.path.isdir(src_dir):
        print("[Err]: invalid src dir: {:s}"
              .format(src_dir))
        exit(-1)

    f_paths = []
    find_files(src_dir, f_paths, ext)
    print("[Info]: find total {:d} files of {:s}"
          .format(len(f_paths), ext))

    ret_f_paths = []
    for f_path in f_paths:
        if np.random.random() > ratio:
            continue
        # img = cv2.imread(f_path, cv2.IMREAD_COLOR)
        # h, w, c = img.shape
        ret_f_paths.append(f_path)
    return ret_f_paths


def random_crop_imgs(img_path_list,
                     dst_dir_path,
                     crop_size=(512, 512)):
    """
    @param img_path_list:
    @param dst_dir_path:
    @param crop_size:
    @return:
    """
    if not isinstance(img_path_list, list):
        print("[Err]: input is not a list!")
        exit(-1)

    def random_crop(img, crop_size=(128, 128)):
        """
        @param img:
        @param crop_size:
        @return:
        """
        h, w = img.shape[:2]
        new_w, new_h = crop_size

        if h <= 10 or w <= 10 \
                or h - new_h <= 0 \
                or w - new_w <= 0:
            return None

        y = np.random.randint(0, h - new_h)
        x = np.random.randint(0, w - new_w)
        img = img[y:y + new_h, x:x + new_w, :]
        return img

    dst_dir_path = os.path.abspath(dst_dir_path)
    if not os.path.isdir(dst_dir_path):
        try:
            os.makedirs(dst_dir_path)
        except Exception as e:
            print(e)
        else:
            print("[Info]: {:s} made".format(dst_dir_path))

    cnt = 0
    with tqdm(total=len(img_path_list)) as p_bar:
        for img_path in img_path_list:
            if not os.path.isfile(img_path):
                print("[Warning]: invalid img path: {:s}"
                      .format(img_path))
                p_bar.update()
                continue

            img_name = os.path.split(img_path)[-1]
            img_ext = "." + img_name.split(".")[-1]

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            h, w, c = img.shape
            roi = random_crop(img, crop_size)
            if roi is None:
                print("\n[Warning]: invalid img shape: {:d}×{:d}\n".format(w, h))
                p_bar.update()
                continue

            if img_ext == ".jpg":
                save_path = dst_dir_path + "/roi_{:04d}.jpg".format(cnt)
                cv2.imencode(".jpg", roi, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tofile(save_path)
            elif img_ext == ".png":
                save_path = dst_dir_path + "/roi_{:04d}.png".format(cnt)
                cv2.imencode(".png", roi, [cv2.IMWRITE_PNG_COMPRESSION, 0])[1].tofile(save_path)

            cnt += 1
            p_bar.update()


def run_random_sample_and_crop_imgs(src_img_dir,
                                    dst_img_dir,
                                    ext=".jpg",
                                    crop_size=(448, 448),
                                    ratio=0.001):
    """
    @param src_img_dir:
    @param dst_img_dir:
    @param ext:
    @param crop_size: [w, h]
    @param ratio:
    @return:
    """
    src_img_dir = os.path.abspath(src_img_dir)
    if not os.path.isdir(src_img_dir):
        print("[Err]: invalid img root dir: {:s}"
              .format(src_img_dir))
        exit(-1)

    dst_img_dir = os.path.abspath(dst_img_dir)
    if os.path.isdir(dst_img_dir):
        shutil.rmtree(dst_img_dir)
    try:
        os.makedirs(dst_img_dir)
    except Exception as e:
        print(e)
        exit(-1)

    print("[Info]: crop size: {:d}×{:d}".format(crop_size[0], crop_size[1]))
    img_paths = random_sample_imgs(src_img_dir, ext, ratio)
    random_crop_imgs(img_paths, dst_img_dir, crop_size)


def run_random_sample_imgs(src_root,
                           dst_dir,
                           ext=".jpg",
                           ratio=0.0001):
    """
    @param src_root:
    @param dst_dir:
    @param ext:
    @return:
    """
    src_root = os.path.abspath(src_root)
    if not os.path.isdir(src_root):
        print("[Err]: invalid src root: {:s}"
              .format(src_root))
        exit(-1)

    dst_dir = os.path.abspath(dst_dir)
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    try:
        os.makedirs(dst_dir)
    except Exception as e:
        print(e)
        exit(-1)

    # f_paths = random_sample_imgs(src_root, ext, ratio)

    # -----
    f_paths = []
    find_files(src_root, f_paths, ext)
    print("[Info]: find total {:d} files of [{:s}]"
          .format(len(f_paths), ext))

    cnt = 0
    for f_path in f_paths:
        if np.random.random() > ratio:
            continue

        f_name = os.path.split(f_path)[-1]
        dst_f_path = os.path.abspath(dst_dir + "/" + f_name)
        if not os.path.isfile(dst_f_path):
            shutil.copy(f_path, dst_dir)
            print("--> {:s} [cp to] {:s}".format(f_name, dst_dir))
        cnt += 1
    print("[Info]: total {:d} samples generated".format(cnt))


def filter_imgs(img_dir):
    """
    @param img_dir:
    @return:
    """

    def save_img_uncompressed(save_path, img):
        """
        @param save_path:
        @param img:
        @return:
        """
        f_name = os.path.split(save_path)[-1]
        f_ext = f_name.split(".")[-1]
        if f_ext == "png":
            cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])[1].tofile(save_path)
        elif f_ext == "jpg":
            cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tofile(save_path)
        else:
            print("[Warning]: invalid file type: {:s}".format(f_ext))
            cv2.imwrite(save_path, img)

    img_dir = os.path.abspath(img_dir)
    if not os.path.isdir(img_dir):
        print("[Err]: invalid img dir: {:s}"
              .format(img_dir))
        exit(-1)

    cnt = 0
    f_names = [x for x in os.listdir(img_dir)]
    with tqdm(total=len(f_names)) as p_bar:
        for f_name in f_names:
            f_path = os.path.abspath(img_dir + "/" + f_name)
            if not os.path.isfile(f_path):
                print("\n[Warning]: invalid file path: {:s}\n"
                      .format(f_path))
                p_bar.update()
                continue

            img = cv2.imread(f_path, cv2.IMREAD_COLOR)
            if img is None:
                print("\n[Warning]: read img failed\n".format(f_name))
                p_bar.update()
                continue

            h, w, c = img.shape
            if w < 1000 or w > 2560 or h < 512:
                os.remove(f_path)
                print("\n--> {:s} removed, img size: {:d}×{:d}\n"
                      .format(f_path, w, h))
                p_bar.update()
                continue
            if h == 1120 or h == 1128 or h == 1480:
                img = img[40:, :, :]
                save_img_uncompressed(f_path, img)
                print("\n--> {:s} cropped\n".format(f_path))
            elif h == 1088:
                img = img[8:, :, :]
                save_img_uncompressed(f_path, img)
                print("\n--> {:s} cropped\n".format(f_path))

            cnt += 1
            p_bar.update()

    print("[Info]: total {:d} files remained".format(cnt))


def filter_and_copy(src_dir, dst_dir, ext=".jpg"):
    """
    @param src_dir:
    @param dst_dir:
    @param ext:
    @return:
    """
    src_dir = os.path.abspath(src_dir)
    if not os.path.isdir(src_dir):
        print("[Err]: invalid src dir: {:s}".format(src_dir))
        exit(-1)

    dst_dir = os.path.abspath(dst_dir)
    if not os.path.isdir(dst_dir):
        try:
            os.makedirs(dst_dir)
        except Exception as e:
            print(e)
            exit(-1)

    src_f_paths = []
    find_files(src_dir, src_f_paths, ext)
    print("[Info]: find total {:d} files of [{:s}]"
          .format(len(src_f_paths), ext))

    for src_f_path in src_f_paths:
        img = cv2.imread(src_f_path, cv2.IMREAD_COLOR)
        f_name = os.path.split(src_f_path)[-1]
        if img is None:
            print("\n[Warning]: read img failed\n".format(f_name))
            continue

        h, w, c = img.shape
        if w < 1000 or w > 2560 or h < 512:
            continue
        else:
            dst_f_path = os.path.abspath(dst_dir + "/{:s}".format(f_name))
            if not os.path.isfile(dst_f_path):
                shutil.copyfile(src_f_path, dst_f_path)
                print("=> {:s} [cp to] {:s}".format(f_name, dst_dir))

def filter_img_pairs(lq_dir, hq_dir, ext=".jpg"):
    """
    @param lq_dir:
    @param hq_dir:
    @param ext:
    @return:
    """
    lq_dir = os.path.abspath(lq_dir)
    if not os.path.isdir(lq_dir):
        print("[Err]: invalid lq dir: {:s}"
              .format(lq_dir))
        exit(-1)

    hq_dir = os.path.abspath(hq_dir)
    if not os.path.isdir(hq_dir):
        print("[Err]: invalid hq dir: {:s}"
              .format(hq_dir))
        exit(-1)

    lq_f_paths, hq_f_paths = [], []
    find_files(lq_dir, lq_f_paths, ext)
    find_files(hq_dir, hq_f_paths, ext)
    lq_f_names = [os.path.split(x)[-1] for x in lq_f_paths]
    hq_f_names = [os.path.split(x)[-1] for x in hq_f_paths]

    if len(hq_f_names) > len(lq_f_names):  # rm redundant hq files
        for hq_path, hq_name in zip(hq_f_paths, hq_f_names):
            if hq_name in lq_f_names:
                continue
            os.remove(hq_path)
            print("--> rm {:s}".format(hq_name))

    elif len(lq_f_names) > len(hq_f_names):
        for lq_path, lq_name in zip(lq_f_paths, lq_f_names):
            if lq_name in hq_f_names:
                continue
            os.remove(lq_path)
            print("--> rm {:s}".format(lq_path))


def add_prefix(dir_path, prefix, ext=".jpg"):
    """
    @param dir_path:
    @param prefix:
    @param ext:
    @return:
    """
    dir_path = os.path.abspath(dir_path)
    if not os.path.isdir(dir_path):
        print("[Err]: invalid dir path: {:s}"
              .format(dir_path))
        exit(-1)

    file_paths = []
    find_files(dir_path, file_paths, ext=ext)
    for f_path in file_paths:
        f_name = os.path.split(f_path)[-1]
        # base_name, suffix = f_name.split(".")
        # new_name = prefix + "_" + base_name + ".{:s}".format(suffix)
        new_name = prefix + "_" + f_name
        new_path = os.path.abspath(dir_path + "/" + new_name)
        os.rename(f_path, new_path)
        print("--> {:s} [rename to] {:s}\n".format(f_path, new_path))


def cp_img_pairs(src_root, dst_root, ext=".jpg"):
    """
    @param src_root:
    @param dst_root:
    @param ext:
    @return:
    """
    src_root = os.path.abspath(src_root)
    if not os.path.isdir(src_root):
        print("[Err]: invalid src root: {:s}"
              .format(src_root))
        exit(-1)

    dst_root = os.path.abspath(dst_root)
    if not os.path.isdir(dst_root):
        print("[Err]: invalid src root: {:s}"
              .format(dst_root))
        exit(-1)

    src_HR_dir = os.path.abspath(src_root + "/HR")
    src_LR_dir = os.path.abspath(src_root + "/LR")
    if not os.path.isdir(src_HR_dir):
        print("[Err]: invalid src HR dir: {:s}"
              .format(src_HR_dir))
        exit(-1)

    if not os.path.isdir(src_LR_dir):
        print("[Err]: invalid src LR dir: {:s}"
              .format(src_LR_dir))
        exit(-1)

    dst_HR_dir = os.path.abspath(dst_root + "/HR")
    dst_LR_dir = os.path.abspath(dst_root + "/LR")
    if not os.path.isdir(dst_HR_dir):
        print("[Err]: invalid dst HR dir: {:s}"
              .format(dst_HR_dir))
        exit(-1)
    if not os.path.isdir(dst_LR_dir):
        print("[Err]: invalid dst LR dir: {:s}"
              .format(dst_LR_dir))
        exit(-1)

    src_HR_paths = []
    src_LR_paths = []
    find_files(src_HR_dir, src_HR_paths, ext)
    find_files(src_LR_dir, src_LR_paths, ext)
    src_HR_paths.sort()
    src_LR_paths.sort()

    assert len(src_HR_paths) == len(src_LR_paths)
    for src_HR_path, src_LR_path in zip(src_HR_paths, src_LR_paths):
        src_HR_name = os.path.split(src_HR_path)[-1]
        src_LR_name = os.path.split(src_LR_path)[-1]
        assert src_HR_name == src_LR_name

        dst_HR_path = os.path.abspath(dst_HR_dir + "/{:s}".format(src_HR_name))
        dst_LR_path = os.path.abspath(dst_LR_dir + "/{:s}".format(src_LR_name))

        if not os.path.isfile(dst_HR_path):
            shutil.copyfile(src_HR_path, dst_HR_path)
            print("=> {:s} [cp to] {:s}"
                  .format(src_HR_name, dst_HR_dir))
        if not os.path.isfile(dst_LR_path):
            shutil.copyfile(src_LR_path, dst_LR_path)
            print("=> {:s} [cp to] {:s}"
                  .format(src_LR_name, dst_LR_dir))


def process_biying_to_DGN(Biying_root,
                          dst_root):
    """
    @param Biying_root:
    @param dst_root:
    @return:
    """
    Biying_root = os.path.abspath(Biying_root)
    if not os.path.isdir(Biying_root):
        print("[Err]: invalid Biying root: {:s}"
              .format(Biying_root))
        exit(-1)

    src_HR_dir = os.path.abspath(Biying_root + "/trainset/HR")
    src_LR_dir = os.path.abspath(Biying_root + "/trainset/LR")
    if not os.path.isdir(src_HR_dir):
        print("[Err]: invalid original path: {:s}".format(src_HR_dir))
        exit(-1)
    if not os.path.isdir(src_LR_dir):
        print("[Err]: invalid bokeh path: {:s}".format(src_LR_dir))
        exit(-1)

    dst_root = os.path.abspath(dst_root)
    dst_HR_dir = os.path.abspath(dst_root + "/HR")
    dst_LR_dir = os.path.abspath(dst_root + "/LR")
    if not os.path.isdir(dst_root):
        try:
            os.makedirs(dst_root)
            print("{:s} made".format(dst_root))

            if not os.path.isdir(dst_HR_dir):
                os.makedirs(dst_HR_dir)
                print("{:s} made".format(dst_HR_dir))

            if not os.path.isdir(dst_LR_dir):
                os.makedirs(dst_LR_dir)
                print("{:s} made".format(dst_LR_dir))

        except Exception as e:
            print(e)

    src_HR_paths = []
    src_LR_paths = []
    find_files(src_HR_dir, src_HR_paths, ext=".jpg")
    find_files(src_LR_dir, src_LR_paths, ext=".jpg")
    src_HR_paths.sort()
    src_LR_paths.sort()

    assert len(src_HR_paths) == len(src_LR_paths)

    for src_HR_path, src_LR_path in zip(src_HR_paths, src_LR_paths):
        HR_name = os.path.split(src_HR_path)[-1]
        LR_name = os.path.split(src_LR_path)[-1]
        assert HR_name == LR_name

        src_HR_img = cv2.imread(src_HR_path, cv2.IMREAD_COLOR)
        src_LR_img = cv2.imread(src_LR_path, cv2.IMREAD_COLOR)
        h_HR, w_HR, c_HR = src_HR_img.shape
        h_LR, w_LR, c_LR = src_LR_img.shape

        # generate LR image
        dst_LR_path = os.path.abspath(dst_LR_dir + "/{:s}".format(LR_name))
        if not os.path.isfile(dst_LR_path):
            shutil.copyfile(src_LR_path, dst_LR_path)
            print("=> {:s} [cp to] {:s}".format(LR_name, dst_HR_dir))

        # generate HR image(equal size of LR)
        LR_size = (w_LR, h_LR)
        idx = np.random.randint(1, 6)
        if idx == 1:
            new_HR_img = cv2.resize(src_HR_img, LR_size, cv2.INTER_NEAREST)
        elif idx == 2:
            new_HR_img = cv2.resize(src_HR_img, LR_size, cv2.INTER_LINEAR)
        elif idx == 3:
            new_HR_img = cv2.resize(src_HR_img, LR_size, cv2.INTER_CUBIC)
        elif idx == 4:
            new_HR_img = cv2.resize(src_HR_img, LR_size, cv2.INTER_AREA)
        elif idx == 5:
            new_HR_img = cv2.resize(src_HR_img, LR_size, cv2.INTER_LANCZOS4)

        dst_HR_path = os.path.abspath(dst_HR_dir + "/{:s}".format(HR_name))
        cv2.imwrite(dst_HR_path, new_HR_img)


if __name__ == "__main__":
    # gen_HQs(img_path_list_f="../files/train_crnn_file_list221230.txt",
    #         HQ_dir="../../../HQ")

    # gen_LQHQ(img_path_list_f="../files/train_crnn_file_list221230.txt",
    #          HQ_dir="../../../img2img/HQ",
    #          LQ_dir="../../../img2img/LQ")

    # filter_HQLQ_pairs(root_dir="../../../img2img/")

    # augment_HQLQ_dataset(src="/mnt/diskd/even/LPDataSingle",
    #                      dst_root="../../../img2img")
    # augment_HQLQ_dataset(src="../config/img2img/train.txt",
    #                      dst_root="../../../img2img")

    # gen_lost_LQs(root_dir="../../../img2img")
    # filter_HQLQ_pairs(root_dir="../../../img2img/")

    # unify_HQ_LQ_imgsize(root_dir="../../../img2img")
    # filter_HQLQ_pairs(root_dir="../../../img2img/")

    # gen_lost_LQs(root_dir="../../../img2img")

    # cvt_png_to_jpg(src_dir="../../../img2img/HQ")

    # unify_HQ_LQ_imgsize(root_dir="../../../img2img")

    # gen_val_set(data_root="../../../img2img/",
    #             ratio=0.0020)

    # viz_img_bursts(root_dir="../../results/img2img/ir-sde",
    #                out_dir="/mnt/diske/vis_plate_gen_3",
    #                num_src_dirs=5,
    #                ext=".png")

    # add_specify_plates_to_test(train_root="../../../img2img",
    #                            test_root="../../../img2img/val")

    # viz_txt2img_set(src_dir="../../results/img2img/img_translate",
    #                 viz_dir="/mnt/diske/vis_plate_gen_5")

    # ----------
    # rename_LPs(root_dir="../../../img2img/")
    # gen_lost_LQs(root_dir="../../../img2img")
    # filter_HQLQ_pairs(root_dir="../../../img2img/")
    # split_and_statistics(root_dir="../../../img2img/")
    # gen_lost_LQs(root_dir="../../../img2img")

    # rename_no_suffix_files(dir_path="../../../img2img/HQ")
    # gen_lost_LQs(root_dir="../../../img2img")
    # filter_HQLQ_pairs(root_dir="../../../img2img/")
    # split_and_statistics(root_dir="../../../img2img/")

    # run_parsing_plates(src_root_dir="/mnt/diske/lyw/NewlyLabeledImages",
    #                    dst_root_dir="/mnt/diske/lyw/NewlyParsedImages")

    # cp_files(src_dir="/mnt/diske/lyw/NewlyParsedImages",
    #          dst_dir="../../../img2img/HQ")
    # rename_no_suffix_files(dir_path="../../../img2img/HQ")
    # gen_lost_LQs(root_dir="../../../img2img")
    # filter_HQLQ_pairs(root_dir="../../../img2img/")
    # split_and_statistics(root_dir="../../../img2img/")

    # ---------
    # parsePlates(src_dir="/mnt/diske/lyw/NewlyLabeledImages/hei",
    #             dst_dir="/mnt/diske/lyw/NewlyParsedImages/hei",
    #             clock_wise=False)
    # cp_files(src_dir="/mnt/diske/lyw/NewlyParsedImages/hei",
    #          dst_dir="../../../img2img/HQ")
    # gen_lost_LQs(root_dir="../../../img2img")
    # filter_HQLQ_pairs(root_dir="../../../img2img/")
    # split_and_statistics(root_dir="../../../img2img/")

    # ----------
    # cp_files(src_dir="/mnt/diske/lyw/ToBeCheckedPlates/00010",
    #          dst_dir="../../../img2img/HQ")
    # gen_lost_LQs(root_dir="../../../img2img")
    # filter_HQLQ_pairs(root_dir="../../../img2img/")
    # split_and_statistics(root_dir="../../../img2img/")

    # ----------
    # run_random_sample_and_crop_imgs(src_img_dir="/mnt/diske/Picture_data",
    #                                 dst_img_dir="/mnt/diske/ROIs",
    #                                 ext=".jpg",
    #                                 ratio=0.0001)

    # run_random_sample_and_crop_imgs(src_img_dir="/mnt/diske/Picture_data",
    #                                 dst_img_dir="/mnt/diske/ROIs",
    #                                 ext=".jpg",
    #                                 crop_size=(1024, 1024),
    #                                 ratio=0.0001)

    # -----
    # run_random_sample_imgs(src_root="/mnt/diske/Picture_data",
    #                        dst_dir="/mnt/diske/RandomSamples",
    #                        ext=".jpg",
    #                        ratio=0.005)
    # filter_imgs(img_dir="/mnt/diske/RandomSamples_2")

    # -----
    # filter_img_pairs(lq_dir="/mnt/ssd/lyw/SISR_data/LR/X2",
    #                  hq_dir="/mnt/ssd/lyw/SISR_data/HR",
    #                  ext=".jpg")

    # add_prefix(dir_path="/mnt/diske/DPED dataset/dped/sony/training_data/HR",
    #            prefix="sony")
    # cp_img_pairs(src_root="/mnt/diske/DPED dataset/dped/sony/training_data",
    #              dst_root="/mnt/ssd/lyw/img2img_data/DGN")

    filter_and_copy(src_dir="/mnt/diske/lyw/BiyingWallPaper_X2/LR",
                    dst_dir="/mnt/diske/RandomSamples_2")

    # process_biying_to_DGN()

    print("--> Done.")
