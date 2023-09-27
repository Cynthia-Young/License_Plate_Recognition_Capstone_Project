import cv2
import numpy as np
import json

MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最大面积


# 读取图片文件
def imreadex(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)

# 像素坐标限制
def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0

# 掩膜计算
def change_color(img):
    height = img.shape[0]
    width = img.shape[1]

    # 设定阈值
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    lower_yellow = np.array([15, 55, 55])
    upper_yellow = np.array([50, 255, 255])
    lower_green = np.array([0, 3, 116])
    upper_green = np.array([76, 211, 255])


    # 转换为HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 根据阈值构建掩膜
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # cv2.imshow('mask_blue', mask_blue)
    # cv2.waitKey()
    # cv2.imshow('mask_yellow', mask_yellow)
    # cv2.waitKey()
    # cv2.imshow('mask_green', mask_green)
    # cv2.waitKey()

    # 对原图像和掩膜进行位运算
    res_blue = cv2.bitwise_and(img, img, mask=mask_blue)
    res_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)
    res_green = cv2.bitwise_and(img, img, mask=mask_green)

    '''
    cv2.imshow('res_blue', res_blue)
    cv2.waitKey()
    cv2.imshow('res_yellow', res_yellow)
    cv2.waitKey()
    cv2.imshow('res_green', res_green)
    cv2.waitKey()
    '''


    return res_blue, res_yellow, res_green

# 预处理图片
class preprocess:
    def __init__(self):
        # 车牌识别的部分参数保存在js中，便于根据图片分辨率做调整
        f = open('config.js')
        j = json.load(f)
        for c in j["config"]:
            if c["open"]:
                self.cfg = c.copy()
                break
        else:
            raise RuntimeError('没有设置有效配置参数')

    def preprocess_pic(self, car_pic, resize_rate=1):
        if type(car_pic) == type(""):
            img = imreadex(car_pic)
        else:
            img = car_pic
        pic_hight, pic_width = img.shape[:2]
        #cv2.imshow("before", img)

        # 调整图片尺寸
        if pic_width > MAX_WIDTH:
            pic_rate = MAX_WIDTH / pic_width
            img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * pic_rate)), interpolation=cv2.INTER_LANCZOS4)

        if resize_rate != 1:
            img = cv2.resize(img, (int(pic_width * resize_rate), int(pic_hight * resize_rate)),
                         interpolation=cv2.INTER_LANCZOS4)
            pic_hight, pic_width = img.shape[:2]
        #cv2.imshow("after resize", img)

        print("h,w:", pic_hight, pic_width)
        # res_blue, res_yellow, res_green = change_color(img)

        blur = self.cfg["blur"]
        # 高斯去噪
        if blur > 0:
            img = cv2.GaussianBlur(img, (blur, blur), 0)  # 图片分辨率调整
        oldimg = img
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("after blur", img)
        # cv2.waitKey()

        # res_blue, res_yellow, res_green = change_color(img)

        contours = self.get_con(img)

        print('len(contours)', len(contours))
        # 一一排除不是车牌的矩形区域
        car_contours = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            area_width, area_height = rect[1]
            if area_width < area_height:
                area_width, area_height = area_height, area_width
            wh_ratio = area_width / area_height
            # print(wh_ratio)
            # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
            if wh_ratio > 2.5 and wh_ratio < 5:
                car_contours.append(rect)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
            # oldimg = cv2.drawContours(oldimg, [box], 0, (0, 0, 255), 2)
            # cv2.imshow("edge4", oldimg)
            # cv2.waitKey(0)

        print('可能的区域数量：',len(car_contours))#有多少个可能是车牌的矩形

        print("精确定位")
        card_imgs = []
        # 矩形区域可能是倾斜的矩形，需要矫正角度，以便使用颜色定位
        for rect in car_contours:
            if rect[2] > -1 and rect[2] < 1:  # 创造角度，使得左、高、右、低拿到正确的值
                angle = 1
            else:
                angle = rect[2]
            rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)  # 扩大范围，避免车牌边缘被排除

            box = cv2.boxPoints(rect)
            height_point = right_point = [0, 0]
            left_point = low_point = [pic_width, pic_hight]
            for point in box:
                if left_point[0] > point[0]:
                    left_point = point
                if low_point[1] > point[1]:
                    low_point = point
                if height_point[1] < point[1]:
                    height_point = point
                if right_point[0] < point[0]:
                    right_point = point

            if left_point[1] <= right_point[1]:  # 正角度
                new_right_point = [right_point[0], height_point[1]]
                pts2 = np.float32([left_point, height_point, new_right_point])  # 字符只是高度需要改变
                pts1 = np.float32([left_point, height_point, right_point])
                M = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
                point_limit(new_right_point)
                point_limit(height_point)
                point_limit(left_point)
                card_img = dst[int(left_point[1]):int(height_point[1]), int(left_point[0]):int(new_right_point[0])]
                card_imgs.append(card_img)
            # cv2.imshow("card", card_img)
            # cv2.waitKey(0)
            elif left_point[1] > right_point[1]:  # 负角度

                new_left_point = [left_point[0], height_point[1]]
                pts2 = np.float32([new_left_point, height_point, right_point])  # 字符只是高度需要改变
                pts1 = np.float32([left_point, height_point, right_point])
                M = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
                point_limit(right_point)
                point_limit(height_point)
                point_limit(new_left_point)
                card_img = dst[int(right_point[1]):int(height_point[1]), int(new_left_point[0]):int(right_point[0])]
                card_imgs.append(card_img)

        '''
        for image in card_imgs:
            cv2.imshow("card", image)
            cv2.waitKey()
        '''

        return card_imgs

    def preprocess_pic_2(self, car_pic, resize_rate=1):
        if type(car_pic) == type(""):
            img = imreadex(car_pic)
        else:
            img = car_pic
        pic_hight, pic_width = img.shape[:2]
        # cv2.imshow("before", img)

        # 调整图片尺寸
        if pic_width > MAX_WIDTH:
            pic_rate = MAX_WIDTH / pic_width
            img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * pic_rate)), interpolation=cv2.INTER_LANCZOS4)

        if resize_rate != 1:
            img = cv2.resize(img, (int(pic_width * resize_rate), int(pic_hight * resize_rate)),
                             interpolation=cv2.INTER_LANCZOS4)
            pic_hight, pic_width = img.shape[:2]
        # cv2.imshow("after resize", img)

        print("h,w:", pic_hight, pic_width)

        blur = self.cfg["blur"]
        # 高斯去噪
        if blur > 0:
            img = cv2.GaussianBlur(img, (blur, blur), 0)  # 图片分辨率调整
        oldimg = img
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("after blur", img)
        # cv2.waitKey()

        res_blue, res_yellow, res_green = change_color(img)

        # contours = self.get_con(img)
        contours_blue = self.get_con(res_blue)
        contours_yellow = self.get_con(res_yellow)
        contours_green = self.get_con(res_green)

        print(len(contours_blue))
        print(len(contours_yellow))
        print(len(contours_green))

        contours = []
        for con in contours_blue:
            contours.append(con)
        for con in contours_yellow:
            contours.append(con)
        for con in contours_green:
            contours.append(con)

        print('len(contours)', len(contours))
        # 一一排除不是车牌的矩形区域
        car_contours = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            area_width, area_height = rect[1]
            if area_width < area_height:
                area_width, area_height = area_height, area_width
            wh_ratio = area_width / area_height
            # print(wh_ratio)
            # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
            if wh_ratio > 2.5 and wh_ratio < 5:
                car_contours.append(rect)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
            # oldimg = cv2.drawContours(oldimg, [box], 0, (0, 0, 255), 2)
            # cv2.imshow("edge4", oldimg)
            # cv2.waitKey(0)

        print('可能的区域数量：', len(car_contours))  # 有多少个可能是车牌的矩形

        print("精确定位")
        card_imgs = []
        # 矩形区域可能是倾斜的矩形，需要矫正角度，以便使用颜色定位
        for rect in car_contours:
            if rect[2] > -1 and rect[2] < 1:  # 创造角度，使得左、高、右、低拿到正确的值
                angle = 1
            else:
                angle = rect[2]
            # rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)  # 扩大范围，避免车牌边缘被排除
            rect = (rect[0], (rect[1][0] - 5, rect[1][1] - 7), angle)  # 缩小范围，排除无关车牌边缘

            box = cv2.boxPoints(rect)
            height_point = right_point = [0, 0]
            left_point = low_point = [pic_width, pic_hight]
            for point in box:
                if left_point[0] > point[0]:
                    left_point = point
                if low_point[1] > point[1]:
                    low_point = point
                if height_point[1] < point[1]:
                    height_point = point
                if right_point[0] < point[0]:
                    right_point = point

            if left_point[1] <= right_point[1]:  # 正角度
                new_right_point = [right_point[0], height_point[1]]
                pts2 = np.float32([left_point, height_point, new_right_point])  # 字符只是高度需要改变
                pts1 = np.float32([left_point, height_point, right_point])
                M = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
                point_limit(new_right_point)
                point_limit(height_point)
                point_limit(left_point)
                card_img = dst[int(left_point[1]):int(height_point[1]), int(left_point[0]):int(new_right_point[0])]
                card_imgs.append(card_img)
            # cv2.imshow("card", card_img)
            # cv2.waitKey(0)
            elif left_point[1] > right_point[1]:  # 负角度

                new_left_point = [left_point[0], height_point[1]]
                pts2 = np.float32([new_left_point, height_point, right_point])  # 字符只是高度需要改变
                pts1 = np.float32([left_point, height_point, right_point])
                M = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
                point_limit(right_point)
                point_limit(height_point)
                point_limit(new_left_point)
                card_img = dst[int(right_point[1]):int(height_point[1]), int(new_left_point[0]):int(right_point[0])]
                card_imgs.append(card_img)

        '''
        for image in card_imgs:
            cv2.imshow("card", image)
            cv2.waitKey()
        '''
        print(card_imgs)
        return card_imgs

    def get_con(self, img):
        g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 去掉图像中不会是车牌的区域
        kernel = np.ones((20, 20), np.uint8)
        img_opening = cv2.morphologyEx(g_img, cv2.MORPH_OPEN, kernel)  # 开操作:去除较小的明亮区域
        img_opening = cv2.addWeighted(g_img, 1, img_opening, -1, 0)  # 融合
        # 找到图像边缘
        ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_edge = cv2.Canny(img_thresh, 100, 200)
        # 使用开运算和闭运算让图像边缘成为一个整体
        kernel = np.ones((self.cfg["morphologyr"], self.cfg["morphologyc"]), np.uint8)
        img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
        # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
        try:
            contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]
        return contours