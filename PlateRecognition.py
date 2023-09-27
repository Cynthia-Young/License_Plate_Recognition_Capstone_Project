import segmentation
import preprocess
import predict
import cv2
import json
import time
import numpy as np


class Recognition():
    # struct_path
    def __init__(self):
        with open('province.json', 'r', encoding='utf-8') as f:
            self.province = json.load(f)

        # 车牌类型保存在cardtype.json中，便于调整
        with open('cardtype.json', 'r', encoding='utf-8') as f:
            self.cardtype = json.load(f)

        # 字母所代表的地区保存在Prefecture.json中，便于更新
        with open('Prefecture.json', 'r', encoding='utf-8') as f:
            self.Prefecture = json.load(f)


    # pre_process_data
    def pretreatment(self,plate_file_path):
        pro = preprocess.preprocess()
        # imgs = pro.preprocess_pic(plate_file_path)
        self.imgs = pro.preprocess_pic(plate_file_path)

        return self.imgs

    def get_license_plate(self, car_pic):
        result = {}
        print(car_pic)
        self.imgs = self.pretreatment(car_pic)

        # seg_image
        for i in range(len(self.imgs)):
            if len(self.imgs[i]):
                self.candidate_plate_image = self.imgs[i]
                break

        # candidate_plate_image = imgs[0]
        self.chars, self.color = segmentation.edge_seg(self.candidate_plate_image)
        # chars = segmentation.proj_seg(candidate_plate_image)

        # predict_char
        license_result = predict.pred(self.chars)
        license_result[0] = self.province[license_result[0]]

        if license_result:
            result['InputTime'] = time.strftime("%Y-%m-%d %H:%M:%S")
            result['Type'] = self.cardtype[self.color]
            result['Picture'] = self.imgs[0]
            result['Number'] = ''.join(license_result[:2]) + '·' + ''.join(license_result[2:])
            try:
                result['From'] = ''.join(self.Prefecture[license_result[0]][license_result[1]])
            except:
                result['From'] = '未知'
            return result
        else:
            return None

# 测试
if __name__ == '__main__':
    c = Recognition()

    path_prefix = "test/"
    pic_path = "7.jpg"
    plate_file_path = path_prefix + pic_path

    # card_imgs, colors = c.pretreatment('./2.jpg')
    result = c.get_license_plate(plate_file_path)
    # cv2.imshow('card', card_imgs)

    cv2.waitKey(0)