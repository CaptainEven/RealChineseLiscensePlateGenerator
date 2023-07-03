# encoding=utf-8

import random

province = ["桂", "贵", "冀", "吉", "京", "琼", "陕", "苏", "湘",
            "渝", "豫", "藏", "川", "鄂", "甘", "赣", "黑", "沪", "津", "晋",
            "鲁", "蒙", "闽", "宁", "青", "皖", "新", "粤", "云", "浙", "辽"]

number = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

char = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L",
        "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def random_select(data_list):
    return data_list[random.randint(0, len(data_list) - 1)]


def random_char_and_num(char_list, num_list, steps):
    lisense = ""
    char_count = 0
    for i in range(steps):
        if char_count < 2:
            str = random_select(char_list + num_list)
        else:
            str = random_select(num_list)
        if str in char_list:
            char_count += 1
        lisense += str
    return lisense


def random_generate_str(color,
                        layer,
                        yellow_special,
                        white_special,
                        green_special,
                        black_special):
    """
    @param color:
    @param layer:
    @param yellow_special:
    @param white_special:
    @param green_special:
    @param black_special:
    @return:
    """
    color_rand = random_select(color)
    layer_rand = layer[0]
    if color_rand == color[1] or color_rand == color[2]:
        layer_rand = random_select(layer)

    lisense = ""
    if color_rand == "blue":
        lisense = random_select(province) + random_select(char[:-1]) + random_char_and_num(char, number, 5)

    elif color_rand == "yellow":
        yellow_class = random_select(yellow_special)
        if yellow_class == "xue":
            layer_rand = "single"
            lisense = random_select(province) + random_select(char[:-1]) + random_char_and_num(char, number, 4) + "学"

        elif yellow_class == "gua":
            layer_rand = "double"
            lisense = random_select(province) + random_select(char[:-1]) + random_char_and_num(char, number, 4) + "挂"

        else:
            lisense = random_select(province) + random_select(char[:-1]) + random_char_and_num(char, number, 5)

    elif color_rand == "white":
        white_class = random_select(white_special)
        if white_class == "wujing":
            lisense = "WJ" + random_select(province + ["X", "N"]) + random_char_and_num(number, number,
                                                                                        4) + random_select(
                number + ["X", "B"])

        elif white_class == "jun":
            lisense = random_char_and_num(char, char, 2) + random_char_and_num(number, number, 5)

        elif white_class == "jing":
            layer_rand = "single"
            lisense = random_select(province) + random_select(char[:-1]) + random_char_and_num(number, number, 4) + "警"

        elif white_class == "yingji":
            layer_rand = "single"
            lisense = random_select(province) + random_select(["X", "S"]) + random_char_and_num(number, number,
                                                                                                4) + "应急"

    elif color_rand == "black":
        black_class = random_select(black_special)
        if black_class == "gang":
            lisense = "粤Z" + random_char_and_num(char, number, 4) + "港"

        elif black_class == "ao":
            lisense = "粤Z" + random_char_and_num(char, number, 4) + "澳"

        elif black_class == "dashi":
            if random.randint(0, 1) == 0:
                lisense = "使" + random_char_and_num(number, number, 6)
            else:
                lisense = random_char_and_num(number, number, 6) + "使"

        elif black_class == "lingshi":
            lisense = random_select(province) + random_select(["A"] + number) + random_char_and_num(number, number,
                                                                                                    4) + "领"

    elif color_rand == "green":
        green_class = random_select(green_special)
        if green_class == "bus":
            lisense = random_select(province) + random_select(char[:-1]) + random_char_and_num(number, number,
                                                                                               5) + random_select(
                char[:10])

        else:
            lisense = random_select(province) + random_select(char[:-1]) + random_select(
                char[:10]) + random_char_and_num(char, number, 1) + random_char_and_num(number, number, 4)

    lisense_chars = lisense + "_" + color_rand + "_" + layer_rand
    print(lisense_chars)
    return lisense_chars


if __name__ == "__main__":
    color = ["blue", "yellow", "white", "black", "green"]
    layer = ["single", "double"]
    yellow_special = ["xue", "gua", "normal"]
    white_special = ["wujing", "jun", "jing", "yingji"]
    green_special = ["bus", "normal"]
    black_special = ["gang", "ao", "dashi", "lingshi"]

    out_file = "/mnt/ssd/sunshangyun/plate_char/random_gen_file_list.txt"
    out = open(out_file, "w")
    counts = 200
    for i in range(counts):
        img_name = random_generate_str(color, layer, yellow_special, white_special, green_special, black_special)
        out.write(img_name + '\n')
    out.close()
