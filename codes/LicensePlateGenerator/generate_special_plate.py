# encoding=utf-8

import argparse
import os
import glob
import cv2
from LicensePlateGenerator.generate_multi_plate import MultiPlateGenerator


###################################根据现场真实车牌生成制式车牌###################################
#
# if __name__ == '__main__':
#     # data_path = "/home/wuhan/sunshangyun/data/all_provinces_plate_test_200_unity_add/data_for_gan/char_gan/rec_correct"
#     # write_path = "/home/wuhan/sunshangyun/data/all_provinces_plate_test_200_unity_add/data_for_gan/char_gan/lisense_plate_rec_correct"
#     #
#     # out_file_list = "/home/wuhan/sunshangyun/data/all_provinces_plate_test_200_unity_add/data_for_gan/char_gan/lisense_plate_rec_correct/file_list.txt"
#
#     data_path = "/users/sunshangyun/data/data_for_gan/special_province/plate_b/train1019"
#     write_path = "/users/sunshangyun/data/data_for_gan/special_province/plate_a/train1019"
#     out_file_list = "/users/sunshangyun/data/data_for_gan/special_province/file_list1019.txt"
#
#     # data_path = "/home/wuhan/sunshangyun/data/zhuanli/plate_b/"
#     # write_path = "/home/wuhan/sunshangyun/data/zhuanli/plate_a/"
#     # out_file_list = "/home/wuhan/sunshangyun/data/zhuanli/plate_a/file_list1019.txt"
#
#
#     out_put = open(out_file_list, "w")
#
#
#     province_char = ["桂", "贵", "冀", "吉", "京", "琼", "陕", "苏", "湘", "渝", "豫",
#                     "藏", "川", "鄂", "甘", "赣", "黑", "沪", "津", "晋", "鲁", "蒙", "闽", "宁",
#                     "青", "皖", "新", "粤", "云", "浙", "辽", "港", "澳"]
#
#     provinces = {}
#     provinces_yellow_single = {}
#     provinces_yellow_double = {}
#     provinces_green_single = {}
#     for char in province_char:
#         provinces[char] = 0
#         provinces_yellow_single[char] = 0
#         provinces_yellow_double[char] = 0
#         provinces_green_single[char] = 0
#
#     generator = MultiPlateGenerator('plate_model', 'font_model')
#
#     path = os.path.join(data_path, '*.jpg')
#     images_path = glob.glob(path)
#     images_path = sorted(images_path)
#
#     for image_path in images_path:
#         image_name = image_path.split("/")[-1]
#         args = image_name[:-4].split("_")
#         plate_number = args[0]
#         bg_color = args[1]
#         double = args[2]
#
#         if bg_color == "green":
#             bg_color = "green_car"
#
#         if bg_color == "greenBus":
#             bg_color = "green_truck"
#
#         if double == "double":
#             double = 1
#         else:
#             double = 0
#
#         final_path = os.path.join(write_path, image_name)
#
#         print("img_name:" + str(image_name))
#         if "~" in image_name or (bg_color == "yellow" and len(plate_number) == 8):
#             continue
#
#         out_put.write(final_path + "\n")
#
#         if plate_number[0] in province_char:
#             key = plate_number[0]
#             if plate_number[-1] in province_char:
#                 key = plate_number[-1]
#             provinces[key] += 1
#             if "yellow_single" in image_name:
#                 provinces_yellow_single[key] += 1
#             elif "yellow_double" in image_name:
#                 provinces_yellow_double[key] += 1
#             elif "green" in image_name:
#                 provinces_green_single[key] += 1
#
#         img = generator.generate_plate_special(plate_number, bg_color, double)
#
#         cv2.imwrite(final_path, img)
#
#     out_put.close()
#     print("provinces:" + str(sorted(provinces.items(), key=lambda kv:(kv[1], kv[0]))))
#     print("provinces_yellow_single:" + str(sorted(provinces_yellow_single.items(), key=lambda kv:(kv[1], kv[0]))))
#     print("provinces_yellow_double:" + str(sorted(provinces_yellow_double.items(), key=lambda kv:(kv[1], kv[0]))))
#     print("provinces_green_single:" + str(sorted(provinces_green_single.items(), key=lambda kv:(kv[1], kv[0]))))


##################################指定字符串、颜色、单双层生成车牌##############################

def generate_one_plate(generator, plate_number, bg_color, double, final_path):
    """
    @param generator:
    @param plate_number:
    @param bg_color:
    @param double:
    @param final_path:
    @return:
    """
    if double == 0:
        img_name = str(plate_number) + "_" + str(bg_color) + "_single"
    else:
        img_name = str(plate_number) + "_" + str(bg_color) + "_double"
    img = generator.generate_plate_special(plate_number, bg_color, double)
    final_path = os.path.join(final_path, img_name + ".jpg")
    print(final_path)
    cv2.imwrite(final_path, img)


if __name__ == '__main__':
    plate_number = "云L846GA"
    bg_color = "blue"
    double = 0
    generator = MultiPlateGenerator('plate_model', 'font_model')
    final_path = "/users/sunshangyun/data/"
    generate_one_plate(generator, plate_number, bg_color, double, final_path)
