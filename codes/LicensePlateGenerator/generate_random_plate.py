from generate_multi_plate import MultiPlateGenerator
import cv2


##################随机生成训练数据集#####################


out_dir = "/users/sunshangyun/LicensePlateGenerator/blue_single/"
multiplategenerator = MultiPlateGenerator('plate_model', 'font_model')
i = 0
while i < 50000:
    img_plate_model, number_xy, plate_number, bg_color, is_double = multiplategenerator.generate_plate()

    plate_model = "double"
    if is_double == False:
        plate_model = "single"

    #指定颜色
    print("plate_model =="+str(plate_model) + " bg_color =="+str(bg_color))

    # if plate_model == "single" and bg_color == "green_car" :
    img_name = str(plate_number)+"_"+str(bg_color)+"_"+str(plate_model)+".jpg"
    cv2.imwrite(out_dir + img_name, img_plate_model)
    i +=1