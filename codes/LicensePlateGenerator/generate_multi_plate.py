# encoding=utf-8

import argparse
import os
import shutil
import sys
from glob import glob

import cv2
import numpy as np

sys.path.append('.')

from LicensePlateGenerator.plate_number import generate_plate_number_black_gangao, \
    generate_plate_number_black_shi, \
    generate_plate_number_black_ling
from LicensePlateGenerator.plate_number import generate_plate_number_blue, \
    generate_plate_number_yellow_gua
from LicensePlateGenerator.plate_number import letters, digits
from LicensePlateGenerator.plate_number import random_select, \
    generate_plate_number_white, generate_plate_number_yellow_xue, generate_plate_number_white_yj

from tqdm import tqdm

plate_char = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F",
              "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W",
              "X", "Y", "Z", "桂", "贵", "冀", "吉", "京", "琼", "陕", "苏", "湘", "渝", "豫",
              "藏", "川", "鄂", "甘", "赣", "黑", "沪", "津", "晋", "鲁", "蒙", "闽", "宁",
              "青", "使", "皖", "新", "粤", "云", "浙", "辽", "军", "空", "兰", "广", "海",
              "成", "应", "急", "学", "警", "港", "澳", "赛", "领", "挂", "WJ", "?", "*", "*", "*"]


def get_position_data(length=7, split_id=1, height=140):
    """
    获取车牌号码在底牌中的位置
    length: 车牌字符数，7或者8, 7为普通车牌、8为新能源车牌(或应急...)
    split_id: 分割空隙
    height: 车牌高度，对应单层和双层车牌
    plate_class:车牌种类，例如wj车牌和应急车牌的字符位置与尺寸与普通蓝绿白车牌不一样
    """
    # 字符位置
    location_xy = np.zeros((length, 4), dtype=np.int32)  # 字符的左上角和右下角的坐标

    if length == 9:  # WJ车牌只有7字符
        location_xy = np.zeros((7, 4), dtype=np.int32)

    # 单层车牌高度
    if height == 140:
        # 单层车牌，y轴坐标固定
        location_xy[:, 1] = 25
        location_xy[:, 3] = 115

        # 螺栓间隔
        step_split = 34 if length == 7 else 49

        # 字符间隔
        step_font = 12 if length == 7 else 9

        # 字符宽度
        width_font = 45

        # WJ车牌单独处理
        if length == 9:
            step_split = 43
            width_font = 84
            step_font = 6

        for i in range(length):
            if length == 9 and i > 6:
                break

            if i == 0:
                location_xy[i, 0] = 15
            elif i == split_id:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_split
            else:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_font

            # 新能源和应急车牌，第一位宽度45，剩下的43。
            if length == 8 and i > 0:
                width_font = 43

            # WJ车牌
            if length == 9 and i > 0:
                width_font = 48

            location_xy[i, 2] = location_xy[i, 0] + width_font
    else:  # 双层
        # 双层车牌第一层
        location_xy[0, :] = [110, 15, 190, 75]
        location_xy[1, :] = [250, 15, 330, 75]

        # WJ车牌单独处理
        if length == 9:
            location_xy[0, :] = [98, 22, 180, 88]
            location_xy[1, :] = [266, 22, 340, 88]

        # 第二层
        width_font = 65
        step_font = 15

        for i in range(2, length):
            if length == 9 and i > 6:  # WJ车牌只有7个字符
                break

            location_xy[i, 1] = 90
            location_xy[i, 3] = 200

            if length == 9:  # 双层wj
                location_xy[i, 1] = 106

            if i == 2:
                location_xy[i, 0] = 27
            else:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_font
            location_xy[i, 2] = location_xy[i, 0] + width_font

    return location_xy


# 字符贴上底板
def copy_to_image_multi(img, font_img, bbox, bg_color, is_red):
    """
    :param img:
    :param font_img:
    :param bbox:
    :param bg_color:
    :param is_red:
    :return:
    """
    x1, y1, x2, y2 = bbox
    font_img = cv2.resize(font_img, (x2 - x1, y2 - y1))
    img_crop = img[y1: y2, x1: x2, :]

    if is_red:
        img_crop[font_img < 200, :] = [0, 0, 255]
    elif 'blue' in bg_color or 'black' in bg_color:
        img_crop[font_img < 200, :] = [255, 255, 255]
    else:
        img_crop[font_img < 200, :] = [0, 0, 0]

    return img


class MultiPlateGenerator:
    def __init__(self, adr_plate_model, adr_font, blue_prob=2.0):
        """
        :param adr_plate_model:
        :param adr_font:
        """
        # 车牌底板路径
        self.adr_plate_model = adr_plate_model
        if not os.path.isdir(self.adr_plate_model):
            print('[Err]: invalid adr_plate_model.')
        print('adr_plate_model: {:s}'.format(os.path.abspath(self.adr_plate_model)))

        # 车牌字符路径
        self.adr_font = adr_font
        if not os.path.isdir(self.adr_font):
            print('[Err]: invalid adr_font.')
        print('adr_font: {:s}'.format(os.path.abspath(self.adr_font)))

        self.blue_prob = blue_prob

        # 车牌字符图片，预存处理
        self.font_imgs = {}
        font_file_names = glob(os.path.join(self.adr_font, '*jpg'))

        for font_file_name in font_file_names:
            font_img = cv2.imdecode(np.fromfile(font_file_name, dtype=np.uint8), 0)

            if '140' in font_file_name:
                font_img = cv2.resize(font_img, (45, 90))
            elif '220' in font_file_name:
                font_img = cv2.resize(font_img, (65, 110))
            elif font_file_name.split('_')[-1].split('.')[0] in letters + digits:
                font_img = cv2.resize(font_img, (43, 90))

            self.font_imgs[os.path.basename(font_file_name).split('.')[0]] = font_img

        # 字符位置
        self.pos_xys = {}
        for i in [7, 8, 9]:
            for j in [1, 2, 4]:  # split id
                for k in [140, 220]:
                    self.pos_xys['{}_{}_{}'.format(i, j, k)] = get_position_data(length=i, split_id=j, height=k)

        print('License generator inited.')

    # 获取字符位置
    def get_location_multi(self, plate_number, height=140):
        """
        :param plate_number:
        :param height:
        :return:
        """
        length = len(plate_number)
        if 'WJ' in plate_number and plate_number[0] == "W":
            length = 9

        if '警' in plate_number:
            split_id = 1
        elif '使' in plate_number:
            split_id = 4
        elif '应急' in plate_number:
            split_id = 1
        else:
            split_id = 2

        loc = None
        try:
            loc = self.pos_xys['{}_{}_{}'.format(length, split_id, height)]
        except Exception as e:
            print(e)

        return loc

    # 随机生成车牌号码，获取底板颜色、单双层
    def generate_plate_number(self):
        """
        :return:
        """
        rate = np.random.random(1)
        if rate < self.blue_prob:
            # 蓝牌
            # seven_prob = np.random.randint(5, 9)
            seven_prob = np.random.randint(1, 2)
            # eight_prob = np.random.randint(3, 5)
            eight_prob = np.random.randint(0, 1)
            plate_number = generate_plate_number_blue(length=random_select([7] * seven_prob + [8] * eight_prob))
        elif self.blue_prob == -1.0:
            generate_plate_number_funcs = [generate_plate_number_white_yj]

            plate_number = random_select(generate_plate_number_funcs)()
        else:
            ## 白牌、黄牌教练车(学)、黄牌挂车、黑色港澳、黑色使、领馆、应急
            generate_plate_number_funcs = [generate_plate_number_white,
                                           generate_plate_number_yellow_xue,
                                           generate_plate_number_yellow_gua,
                                           generate_plate_number_black_gangao,
                                           generate_plate_number_black_shi,
                                           generate_plate_number_black_ling,
                                           generate_plate_number_white_yj]

            plate_number = random_select(generate_plate_number_funcs)()

        # 车牌底板颜色: 这里控制蓝黄比例
        # bg_color = random_select(['blue'] + ['yellow'])
        bg_color = 'blue' if np.random.random() < self.blue_prob else 'yellow'

        if len(plate_number) == 8:
            if '应急' in plate_number:  # 白色应急
                bg_color = 'white'
            else:  # 新能源
                if plate_number[2] in digits:
                    bg_color = 'green_truck'
                else:
                    bg_color = 'green_car'

                # bg_color = random_select(['green_car'] * 9 + ['green_truck'])
        elif len(set(plate_number) & set(['使', '领', '港', '澳'])) > 0:
            bg_color = 'black'
        elif '警' in plate_number or plate_number[0] in letters:
            bg_color = 'white'
        elif len(set(plate_number) & set(['学', '挂'])) > 0:
            bg_color = 'yellow'

        is_double = random_select([False] * 6 + [True] * 4)

        if '使' in plate_number:
            bg_color = 'black_shi'

        if '挂' in plate_number:
            # 挂车双层
            is_double = True
        elif len(set(plate_number) & set(['使', '领', '港', '澳', '学', '警'])) > 0 \
                or len(plate_number) == 8 or bg_color == 'blue':
            # 使领港澳学警、新能源、蓝色都是单层
            is_double = False

        # 第二个字符验证
        if plate_number[1] in [str(x) for x in range(10)]:  # 第二个字符是数字
            plate_number = plate_number.replace(plate_number[1], random_select(letters[:-15]))
        elif plate_number[1] in letters[-15:]:  # 第二个字符不合理
            plate_number = plate_number.replace(plate_number[1], random_select(letters[:-15]))

        # special，首字符为字母、单层则是军车
        if plate_number[0] in letters and not is_double:
            bg_color = 'white_army'

        return plate_number, bg_color, is_double

    # 随机生成制式车牌图片
    def generate_plate(self, enhance=False):
        """
        :param enhance:
        :return:
        """
        # enhance = True if np.random.random() > 0.95 else False
        plate_number, bg_color, is_double = self.generate_plate_number()

        plate_number = list(plate_number)
        # plate_number[1] = "A"
        # plate_number[2] = "D"
        # plate_number[3] = "D"
        plate_number = "".join(plate_number)

        height = 220 if is_double else 140

        # 获取底板图片
        # print(plate_number, height, bg_color, is_double)
        number_xy = self.get_location_multi(plate_number, height)
        if number_xy is None:
            print('plate number {:s} can not be generated @func get_location_multi.'.format(plate_number))
            return None

        img_plate_model = cv2.imread(os.path.join(self.adr_plate_model, '{}_{}.PNG'.format(bg_color, height)))

        plate_number_length = len(plate_number)
        if plate_number[0] == "W" and plate_number[1] == "J" and height == 220:
            plate_number_length = 7

        img_plate_model = cv2.resize(img_plate_model, (440 if plate_number_length == 7 else 480, height))

        for i in range(len(plate_number)):
            if len(plate_number) == 8:
                # # 新能源
                # font_img = self.font_imgs['green_{}'.format(plate_number[i])]
                if bg_color == 'green':  # 新能源
                    font_img_name = 'green_{}'.format(plate_number[i])
                    font_img = self.font_imgs[font_img_name]
                # else:
                #                 #     # font_img_name = '{}_{}'.format(height, plate_number[i])
                #                 #     font_img_name = 'green_{}'.format(plate_number[i])
                #                 #     if font_img_name in self.font_imgs:
                #                 #         font_img = self.font_imgs[font_img_name]
                #                 #     else:
                #                 #         font_img_name = '{}_{}'.format(height, plate_number[i])
                #                 #         font_img = self.font_imgs[font_img_name]

                elif plate_number[0] == "W" and plate_number[1] == "J":
                    if i == 1:  # wj是一个字符:
                        continue
                    if i == 0:
                        font_img = self.font_imgs["140_WJ"]
                    else:
                        if '{}_{}'.format(height, plate_number[i]) in self.font_imgs:
                            font_img_name = '{}_{}'.format(height, plate_number[i])
                            font_img = self.font_imgs[font_img_name]
                        else:
                            if i < 3:  # 双层中的第1层
                                font_img_name = '220_up_{}'.format(plate_number[i])
                                font_img = self.font_imgs[font_img_name]
                            else:  # 双层中的第2层
                                font_img_name = '220_down_{}'.format(plate_number[i])
                                font_img = self.font_imgs[font_img_name]

                elif bg_color == 'white':
                    font_img_name = '{}_{}'.format(height, plate_number[i])
                    font_img = self.font_imgs[font_img_name]
                else:
                    # font_img_name = '{}_{}'.format(height, plate_number[i])
                    font_img_name = 'green_{}'.format(plate_number[i])
                    font_img = self.font_imgs[font_img_name]
            else:
                if '{}_{}'.format(height, plate_number[i]) in self.font_imgs:
                    font_img = self.font_imgs['{}_{}'.format(height, plate_number[i])]
                else:
                    # 双层车牌字体库
                    if i < 2:
                        font_img = self.font_imgs['220_up_{}'.format(plate_number[i])]
                    else:
                        font_img = self.font_imgs['220_down_{}'.format(plate_number[i])]

            # 字符是否红色
            if (i == 0 and plate_number[0] in letters) or plate_number[i] in ['警', '使', '领']:
                is_red = True
            elif i == 1 and plate_number[0] in letters and np.random.random(1) > 0.8:
                # second letter of army plate
                is_red = True
            elif (i == 0 or i == 2 or i == 7) and plate_char.index(plate_number[i]) > 9 and plate_number[0] == "W" and \
                    plate_number[1] == "J":
                is_red = True
            elif i == 1 and '应急' in plate_number and np.random.random(1) > 0.5:
                is_red = True
            elif i == 6 and '应' in plate_number:
                is_red = True
            elif i == 7 and '急' in plate_number:
                is_red = True
            else:
                is_red = False

            if enhance:
                k = np.random.randint(1, 6)
                kernel = np.ones((k, k), np.uint8)
                if np.random.random(1) > 0.5:
                    font_img = np.copy(cv2.erode(font_img, kernel, iterations=1))
                else:
                    font_img = np.copy(cv2.dilate(font_img, kernel, iterations=1))

            # 贴上底板
            # img_plate_model = copy_to_image_multi(img_plate_model, font_img, number_xy[i, :], bg_color, is_red)

            if plate_number[0] == "W" and plate_number[1] == "J" and i > 1:
                img_plate_model = copy_to_image_multi(img_plate_model,
                                                      font_img,
                                                      number_xy[i - 1, :],
                                                      bg_color,
                                                      is_red)
            else:
                img_plate_model = copy_to_image_multi(img_plate_model,
                                                      font_img,
                                                      number_xy[i, :],
                                                      bg_color,
                                                      is_red)

        img_plate_model = cv2.blur(img_plate_model, (3, 3))

        return img_plate_model, number_xy, plate_number, bg_color, is_double

    def generate_plate_special(self, plate_number, bg_color, is_double, enhance=False):
        """
        生成特定号码、颜色车牌
        :param plate_number: 车牌号码
        :param bg_color: 背景颜色, black/black_shi(使领馆)/
        blue/green_car(新能源轿车)/green_truck(新能源卡车)/white/white_army(军队)/yellow
        :param is_double: 是否双层
        :param enhance: 图像增强
        :return: 车牌图
        """
        if bg_color == 'greencar':
            bg_color = 'green_car'
        elif bg_color == "greenBus":
            bg_color = "green_truck"
        elif bg_color == "green-bus":
            bg_color = "green_truck"
        elif bg_color == 'whitearmy':
            bg_color = 'white_army'

        height = 220 if is_double else 140

        # print(plate_number, height, bg_color, is_double)

        # 字符分布
        loc_xy = self.get_location_multi(plate_number, height)
        if loc_xy is None:
            print('plate number {:s} can not generate.'.format(plate_number))
            return None

        ## ----- select plate model
        if bg_color != 'green' and bg_color != 'white':
            plate_model_name = '{}_{}.PNG'.format(bg_color, height)
        elif bg_color == 'green':  # green car by default
            plate_model_name = '{}_car_{}.PNG'.format(bg_color, height)

        elif bg_color == 'white' and height == 140:
            if '应急' in plate_number:
                plate_model_name = '{}_{}.PNG'.format(bg_color, height)
            elif plate_number[0] == "W" and plate_number[1] == "J":
                plate_model_name = '{}_wj_{}.PNG'.format(bg_color, height)
            else:
                # plate_model_name = '{}_army_{}.PNG'.format(bg_color, height)
                plate_model_name = '{}_{}.PNG'.format(bg_color, height)

        elif bg_color == 'white' and height == 220:
            if plate_number[0] == "W" and plate_number[1] == "J":
                plate_model_name = '{}_wj_{}.PNG'.format(bg_color, height)
            else:
                plate_model_name = '{}_{}.PNG'.format(bg_color, height)

        plate_model_path = self.adr_plate_model + '/' + plate_model_name
        if not os.path.isfile(plate_model_path):
            print('[Warning]: invalid plate model path: {:s}.'.format(plate_model_path))
            return None

        img_plate_model = cv2.imread(plate_model_path)

        plate_number_length = len(plate_number)
        if plate_number[0] == "W" and plate_number[1] == "J" and height == 220:
            plate_number_length = 7

        try:
            img_plate_model = cv2.resize(img_plate_model, (440 if plate_number_length == 7 else 480, height))
        except Exception as e:
            print(e)

        # generate each char of the plate number
        for i in range(len(plate_number)):
            if len(plate_number) == 8:
                if bg_color == 'green':
                    font_img_name = 'green_{}'.format(plate_number[i])
                    font_img = self.font_imgs[font_img_name]
                elif plate_number[0] == "W" and plate_number[1] == "J":
                    if i == 1:  # wj是一个字符:
                        continue
                    if i == 0:
                        font_img = self.font_imgs["140_WJ"]
                    else:
                        if '{}_{}'.format(height, plate_number[i]) in self.font_imgs:
                            font_img_name = '{}_{}'.format(height, plate_number[i])
                            font_img = self.font_imgs[font_img_name]
                        else:
                            if i < 3:  # 双层中的第1层
                                font_img_name = '220_up_{}'.format(plate_number[i])
                                font_img = self.font_imgs[font_img_name]
                            else:  # 双层中的第2层
                                font_img_name = '220_down_{}'.format(plate_number[i])
                                font_img = self.font_imgs[font_img_name]
                elif bg_color == 'white':
                    font_img_name = '{}_{}'.format(height, plate_number[i])
                    font_img = self.font_imgs[font_img_name]
                else:
                    # font_img_name = '{}_{}'.format(height, plate_number[i])
                    font_img_name = 'green_{}'.format(plate_number[i])
                    font_img = self.font_imgs[font_img_name]
            else:
                if '{}_{}'.format(height, plate_number[i]) in self.font_imgs:
                    font_img_name = '{}_{}'.format(height, plate_number[i])
                    font_img = self.font_imgs[font_img_name]
                else:
                    if i < 2:  # 双层中的第1层
                        font_img_name = '220_up_{}'.format(plate_number[i])
                        font_img = self.font_imgs[font_img_name]
                    else:  # 双层中的第2层
                        font_img_name = '220_down_{}'.format(plate_number[i])
                        font_img = self.font_imgs[font_img_name]

            if (i == 0 and plate_number[0] in letters) or plate_number[i] in ['警', '使', '领']:
                is_red = True
            elif (i == 0 or i == 2 or i == 7) and plate_char.index(plate_number[i]) > 9 and plate_number[0] == "W" and \
                    plate_number[1] == "J":
                is_red = True
            elif i == 1 and plate_number[0] in letters and np.random.random(1) > 0.5:
                # second letter of army plate
                is_red = True
            elif i == 1 and '应急' in plate_number and np.random.random(1) > 0.5:
                is_red = True
            elif i == 6 and '应' in plate_number:
                is_red = True
            elif i == 7 and '急' in plate_number:
                is_red = True
            else:
                is_red = False

            if enhance:
                k = np.random.randint(1, 6)
                kernel = np.ones((k, k), np.uint8)
                if np.random.random(1) > 0.5:
                    font_img = np.copy(cv2.erode(font_img, kernel, iterations=1))
                else:
                    font_img = np.copy(cv2.dilate(font_img, kernel, iterations=1))

            if plate_number[0] == "W" and plate_number[1] == "J" and i > 1:
                img_plate_model = copy_to_image_multi(img_plate_model,
                                                      font_img,
                                                      loc_xy[i - 1, :],
                                                      bg_color,
                                                      is_red)
            else:
                img_plate_model = copy_to_image_multi(img_plate_model,
                                                      font_img,
                                                      loc_xy[i, :],
                                                      bg_color,
                                                      is_red)

        # is_double = 'double' if is_double else 'single'
        img_plate_model = cv2.blur(img_plate_model, (3, 3))

        return img_plate_model


def mkdir(path):
    """
    :param path:
    :return:
    """
    try:
        os.makedirs(path)
    except Exception as e:
        print(e)


def parse_LP_args():
    """
    :return:
    """
    parser = argparse.ArgumentParser(description='中国车牌生成器')

    parser.add_argument('--number',
                        type=int,
                        default=207,
                        help='生成车牌数量')
    parser.add_argument('--blue_prob',
                        type=float,
                        default=0.65)

    # '/mnt/diskc/even/DeepMosaic/datasets/plates_synthesis'
    parser.add_argument('--save_dir',
                        type=str,
                        default='/mnt/diskc/even/DeepMosaic/datasets/plates_synth',
                        help='车牌保存路径')
    parser.add_argument('--size',
                        type=tuple,
                        default=(192, 96),  # None | (192, 64) | (192, 96)
                        help='')
    parser.add_argument('--warp_radius',
                        type=float,
                        default=1.8,
                        help='')

    args = parser.parse_args()

    return args


def gen_multi_LPs(save_dir, plate_model_path, font_model_path,
                  number=10000, blue_prob=0.8, size=(192, 96), warp_radius=2.0):
    """
    :param save_dir:
    :param plate_model_path:
    :param font_model_path:
    :param number:
    :param blue_prob:
    :param size:
    :param warp_radius:
    :return:
    """
    # args = parse_LP_args()
    # print(args)

    # 随机生成车牌
    print('Save LPs in {}'.format(save_dir))

    # 清除原有数据
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    else:
        os.makedirs(save_dir)

    mkdir(save_dir)
    generator = MultiPlateGenerator(plate_model_path, font_model_path, blue_prob=blue_prob)

    ## ----- Generating
    for i in tqdm(range(number)):
        ## -----------  generate a license plate
        ret = generator.generate_plate()
        ## -----------

        if ret is None:
            continue

        img, number_xy, gt_plate_number, bg_color, is_double = ret
        if is_double:
            type = 'double'
        else:
            type = 'single'

        bg_color = bg_color.replace("_", "")
        cv2.imwrite(os.path.join(save_dir,
                                 '{}_{}_{}.jpg'
                                 .format(gt_plate_number, bg_color, type)),
                    img)

    ## ----- Resizing and resizing
    if size != None:
        lp_path_list = [save_dir + '/' + x for x in os.listdir(save_dir) if x.endswith('.jpg')]
        print('Total {:d} LPs need to be randomly warpped...'.format(len(lp_path_list)))

        for lp_path in tqdm(lp_path_list):
            if not os.path.isfile(lp_path):
                print('[Err]: invalid LP path.')
                continue

            img = cv2.imread(lp_path, cv2.IMREAD_COLOR)
            h, w, c = img.shape

            ## Do resizing
            if (w, h) == (440, 140) or (w, h) == (480, 140):
                img = cv2.resize(img, (192, 64), cv2.INTER_CUBIC)
            elif (w, h) == (440, 220):
                img = cv2.resize(img, (192, 96), cv2.INTER_CUBIC)

            # # Warpping probability 0.2
            # if warp_radius > 0.0 and np.random.random() < 0.2:
            #     img = warp_img(img, radius=1.8)
            #     # print('{:s} warpped.'.format(lp_path))

            ## save resized(and warpped) LP image
            cv2.imwrite(lp_path, img)

        print('Warping done.')


if __name__ == '__main__':
    gen_multi_LPs(save_dir='/mnt/diskc/even/DeepMosaic/imgs/style',
                  plate_model_path='./plate_model',
                  font_model_path='./font_model',
                  number=200,
                  blue_prob=-1.0)
    print('Done.')

# elif (w, h) == (440, 140) or (w, h) == (480, 140) and args.size == (192, 64):  # resize
#     img = cv2.resize(img, args.size, cv2.INTER_CUBIC)
#     print('{:s} resized to {:d}×{:d}.'.format(lp_path, args.size[0], args.size[1]))
#
#     # Warping probability 0.2
#     if args.warp_radius > 0.0 and np.random.random() < 0.2:
#         img = warp_img(img, radius=1.8)
#         print('{:s} warpped.'.format(lp_path))
#
#     # Save transformed image
#     cv2.imwrite(lp_path, img)
# elif (w, h) == (440, 220) and args.size == (192, 96):
#     img = cv2.resize(img, args.size, cv2.INTER_CUBIC)
#     print('{:s} resized from {:d}×{:d} to {:d}×{:d}.'
#           .format(lp_path, w, h, args.size[0], args.size[1]))
#
#     # Warping probability 0.2
#     if args.warp_radius > 0.0 and np.random.random() < 0.2:
#         img = warp_img(img, radius=1.8)
#         print('{:s} warpped.'.format(lp_path))
#
#     # Save transformed image
#     cv2.imwrite(lp_path, img)
# else:
#     os.remove(lp_path)
#     print('{:s} removed.'.format(lp_path))
#
