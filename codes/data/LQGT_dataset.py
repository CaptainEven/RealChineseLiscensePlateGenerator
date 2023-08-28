# encoding=utf-8

import os
import random
import sys

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data

try:
    to_be_inserted_path = os.path.abspath(".")
    sys.path.insert(0, to_be_inserted_path)
    print("--> to_be_inserted_path: ", to_be_inserted_path)

    to_be_inserted_path = os.path.abspath("..")
    sys.path.insert(0, to_be_inserted_path)
    print("--> to_be_inserted_path: ", to_be_inserted_path)

    import data.util as util
except ImportError:
    pass


class LQGTDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        """
        @param opt:
        """
        super().__init__()

        self.opt = opt
        self.LR_paths, self.GT_paths = None, None
        self.LR_env, self.GT_env = None, None  # environment for lmdb
        self.LR_size, self.GT_size = opt["LR_size"], opt["GT_size"]
        if isinstance(self.GT_size, str):
            self.GT_size = [int(x) for x in self.GT_size.split(",")]
        if isinstance(self.LR_size, str):
            self.LR_size = [int(x) for x in self.LR_size.split(",")]

        # read image list from lmdb or image files
        if opt["data_type"] == "img":
            self.LR_paths = util.get_image_paths(opt["data_type"], opt["dataroot_LQ"])  # LR list
            self.GT_paths = util.get_image_paths(opt["data_type"], opt["dataroot_GT"])  # GT list
        else:
            print("Error: data_type is not matched in Dataset")

        assert self.GT_paths, "Error: GT paths are empty."

        if self.LR_paths and self.GT_paths:
            assert len(self.LR_paths) == len(self.GT_paths), \
                "GT and LR datasets have different number of images - {}, {}." \
                    .format(len(self.LR_paths), len(self.GT_paths))
        self.random_scale_list = [1]

        print("[Info]: dataset size: {:d}".format(len(self.GT_paths)))
        print("GT size:\n", self.GT_size)
        print("LR size:\n", self.LR_size)

    def __getitem__(self, idx):
        """
        @param idx:
        @return:
        """
        GT_path, LR_path = None, None
        scale = 1
        GT_size = tuple(self.GT_size)
        LR_size = tuple(self.LR_size)

        # ---------- get GT image
        GT_path = self.GT_paths[idx]
        LR_path = self.LR_paths[idx]
        GT_name = os.path.split(GT_path)[-1]
        LR_name = os.path.split(LR_path)[-1]
        assert GT_name == LR_name

        # return: Numpy float32, HWC, BGR, [0, 1]
        img_GT = util.read_img(self.GT_env, GT_path, None)
        img_LR = util.read_img(self.LR_env, LR_path, None)

        assert img_GT.shape == img_LR.shape

        # ----- random cropping
        img_GT, (y_min, y_max), (x_min, x_max) = util.random_crop(img_GT, GT_size)

        # ----- cropping according to GT cropping
        img_LR = img_LR[y_min: y_max, x_min: x_max, :]

        H, W, C = img_LR.shape

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        if LR_path is None:
            LR_path = GT_path

        return {
            "LQ": img_LR,
            "GT": img_GT,
            "LQ_path": LR_path,
            "GT_path": GT_path
        }

    def __len__(self):
        """
        @return:
        """
        return len(self.GT_paths)
