"""
generate plate numbers
"""

import cv2
import numpy as np

# 省份
provinces = ["京", "津", "冀", "晋", "蒙", "辽", "吉", "黑", "沪",
             "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘",
             "粤", "桂", "琼", "渝", "川", "贵", "云", "藏", "陕",
             "甘", "青", "宁", "新"]

# provinces = ["湘"]

# "港", "澳", "使", "领", "学", "警", "挂"]
digits = ['{}'.format(x + 1) for x in range(9)] + ['0']

# 英文，没有I、O两个字符
letters = [chr(x + ord('A')) for x in range(26) if not chr(x + ord('A')) in ['I', 'O']]

#绿色车牌第三位和最后一位只能是
letters_green = [chr(x + ord('A')) for x in range(11) if not chr(x + ord('A')) in ['I', 'O']]

# print('letters', digits + letters)

# 随机选取
def random_select(data):
    """
    :param data:
    :return:
    """
    return data[np.random.randint(len(data))]


# 蓝牌   or  绿牌
def generate_plate_number_blue(length=7):
    """
    :param length:
    :return:
    """
    plate = random_select(provinces)

    if length == 8:
        green_id = np.random.randint(2)
        if green_id == 0:  #小新能源
            for i in range(length - 1):
                if i == 0:
                    plate += random_select(letters)
                elif i == 1:
                    plate += random_select(letters_green)
                elif i == 2:
                    plate += random_select(digits + letters)
                else:
                    plate += random_select(digits)
        else:
            for i in range(length - 1):
                if i == 0:
                    plate += random_select(letters)
                elif i == 6:
                    plate += random_select(letters_green)
                else:
                    plate += random_select(digits)

    else:
        #########蓝色车牌最多3个字母#########
        max_letter = 0
        for i in range(length - 1):
            if i == 0:
                plate += random_select(letters)
                max_letter += 1
            else:
                if max_letter > 2:
                    plate += random_select(digits)
                else:
                    select_flag = random_select([0, 1])
                    if select_flag == 0:
                        char = random_select(letters)
                        max_letter += 1
                    else:
                        char = random_select(digits)
                    plate += char

    return plate


# 黄色挂车
def generate_plate_number_yellow_gua():
    """
    :return:
    """
    plate = generate_plate_number_blue()
    return plate[:6] + '挂'


# 教练车
def generate_plate_number_yellow_xue():
    """
    :return:
    """
    plate = generate_plate_number_blue()
    return plate[:6] + '学'


# 白色警车、军车
def generate_plate_number_white(prob=0.3):
    """
    :return:
    """
    plate = generate_plate_number_blue()

    if np.random.random(1) < prob:
        return plate[:6] + '警'
    else:
        first_letter = random_select(letters)
        return first_letter + plate[1:]


def generate_plate_number_white_yj():
    """
    :return:
    """
    plate = generate_plate_number_blue()
    return plate[:6] + '应急'


def generate_plate_number_black_gangao():
    """
    :return:
    """
    plate = generate_plate_number_blue()
    return '粤' + plate[1:6] + random_select(["港", "澳"])


def generate_plate_number_black_ling():
    """
    :return:
    """
    plate = generate_plate_number_blue()
    return plate[:6] + '领'


def generate_plate_number_black_shi():
    """
    :return:
    """
    plate = generate_plate_number_blue()
    return '使' + plate[1:]


def board_bbox(polys):
    """
    :param polys:
    :return:
    """
    x1, y1 = np.min(polys, axis=0)
    x2, y2 = np.max(polys, axis=0)

    return [x1, y1, x2, y2]


def letter_box(img,
               height=608,
               width=1088,
               color=(127.5, 127.5, 127.5)):
    """
    resize while keeping aspect ratio
    resize a rectangular image to a padded rectangular
    :param img: numpy ndarray
    :param height:
    :param width:
    :param color:
    :return:
    """
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])

    # new_shape = [width, height]
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # keep aspect ratio
    dw = (width - new_shape[0])  * 0.5  # width padding
    dh = (height - new_shape[1]) * 0.5  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    # resized, no border
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img,
                             top, bottom, left, right,
                             cv2.BORDER_CONSTANT,
                             value=color)  # padded rectangular

    return img, ratio, dw, dh


def warp_img(img, radius):
    """
    :param img:
    :param radius:
    :return:
    """

    def rand():
        """
        :return:
        """
        return (1 - (-1)) * np.random.random() + (-1)

    if not isinstance(img, np.ndarray):
        print('[Err]: invalid image format.')
        return None

    h, w = img.shape[:2]
    size = (w, h)

    # Build big image
    big_img = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    h_big, w_big = big_img.shape[:2]
    big_img[
    int(0.5 * size[1]):int(1.5 * size[1]),  # height range
    int(0.5 * size[0]):int(1.5 * size[0]),  # width range
    :] = img

    # Build src points and dst points: tl, tr, br, bl
    src_pts = [[0.5 * w, 0.5 * h], [1.5 * w, 0.5 * h], [1.5 * w, 1.5 * h], [0.5 * w, 1.5 * h]]
    dst_pts = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]

    ## ----- Add offset
    offset_mode = np.random.randint(0, 3)
    if offset_mode == 0:
        for i, (src_pt, dst_pt) in enumerate(zip(src_pts, dst_pts)):
            for j, (src_x, dst_x) in enumerate(zip(src_pt, dst_pt)):
                offset = rand() * radius
                src_x += offset

                offset = rand() * radius
                dst_x += offset

                src_pt[j] = src_x
                dst_pt[j] = dst_x

            src_pts[i] = src_pt
            dst_pts[i] = dst_pt

    elif offset_mode == 1:
        for i, src_pt in enumerate(src_pts):
            for j, src_x in enumerate(src_pt):
                offset = rand() * radius
                src_x += offset
                src_pt[j] = src_x

            src_pts[i] = src_pt

    elif offset_mode == 2:
        for i, dst_pt in enumerate(dst_pts):
            for j, dst_x in enumerate(dst_pt):
                offset = rand() * radius
                dst_x += offset
                dst_pt[j] = dst_x

            dst_pts[i] = dst_pt
    ## -----

    ## Warping
    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(big_img, mat, size)

    return warped


if __name__ == '__main__':
    pass
