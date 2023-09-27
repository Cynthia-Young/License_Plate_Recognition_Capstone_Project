import numpy as np
import cv2 as cv
from math import floor

def Square(size,candidate_char_images):
    square_list=[]
    for char in candidate_char_images:
        #背景
        black_img=np.zeros((size,size))
        # 原始图像宽高。
        height, width = char.shape[0], char.shape[1]

        if height>=width:
            # 获得相应等比例的图像宽度。
            width_size = int(width * size / height)
            image_resize = cv.resize(char, (width_size, size))
            w=image_resize.shape[1]
            b1=floor(112-w/2)
            black_img[:,b1:b1+w]=image_resize
        else:
            # 获得相应等比例的图像宽度。
            height_size = int(height * size / width)
            image_resize = cv.resize(char, (size, height_size))
            h= image_resize.shape[0]
            b2 = floor(112 - h / 2)
            black_img[b2:b2+h,:] = image_resize

        square_list.append(black_img)

    return square_list

def judgecolor(img):
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
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # 根据阈值构建掩膜
    mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)  #
    mask_green = cv.inRange(hsv, lower_green, upper_green)  #

    # 对原图像和掩膜进行位运算
    # src1：第一个图像（合并的第一个对象）src2：第二个图像（合并的第二个对象）mask：理解为要合并的规则。
    res_blue = cv.bitwise_and(img, img, mask=mask_blue)
    res_yellow = cv.bitwise_and(img, img, mask=mask_yellow)
    res_green = cv.bitwise_and(img, img, mask=mask_green)

    # 显示图像
    # cv.imshow('frame', img)
    # cv.imshow('mask_blue', mask_blue)
    # cv.imshow('mask_yellow', mask_yellow)
    # cv.imshow('mask_green', mask_green)
    # cv.imshow('res', res)

    # 对mask进行操作--黑白像素点统计  因为不同颜色的掩膜面积不一样
    # 记录黑白像素总和

    blue_white = 0
    blue_black = 0
    yellow_white = 0
    yellow_black = 0
    green_white = 0
    green_black = 0

    # 计算每一列的黑白像素总和
    for i in range(width):
        for j in range(height):
            if mask_blue[j][i] == 255:
                blue_white += 1
            if mask_blue[j][i] == 0:
                blue_black += 1
            if mask_yellow[j][i] == 255:
                yellow_white += 1
            if mask_yellow[j][i] == 0:
                yellow_black += 1
            if mask_green[j][i] == 255:
                green_white += 1
            if mask_green[j][i] == 0:
                green_black += 1

    # print('蓝色--白色 = ', blue_white)
    # print('蓝色--黑色 = ', blue_black)
    # print('黄色--白色 = ', yellow_white)
    # print('黄色--黑色 = ', yellow_black)
    # print('绿色--白色 = ', green_white)
    # print('绿色--黑色 = ', green_black)

    color_list = ['蓝色', '黄色', '绿色']
    num_list = [blue_white, yellow_white, green_white]
    print(num_list.index(max(num_list)))
    print('车牌的颜色为:', color_list[num_list.index(max(num_list))])
    cv.waitKey()
    cv.destroyAllWindows()
    return color_list[num_list.index(max(num_list))]

def Reverse(img):
    height = img.shape[0]
    width = img.shape[1]
    for i in range(0, height):
        for j in range(0, width):
            img[i][j]=255-img[i][j]
    return img

def judge_BinaryImage(img):
    height = img.shape[0]
    width = img.shape[1]
    max_white = 0
    max_black = 0
    for i in range(width):
        white_num = 0
        black_num = 0
        for j in range(height):
            if img[j][i] == 255:
                white_num += 1
            if img[j][i] == 0:
                black_num += 1
        max_white = max(max_white, white_num)
        max_black = max(max_black, black_num)

    if max_white > max_black:  # 白底
        return(Reverse(img))
    else:
        return(img)

def get_grey_binary_image1(candidate_plate_image):
    # 灰度化和二值化该区域
    gray_image = cv.cvtColor(candidate_plate_image, cv.COLOR_BGR2GRAY)
    ret, binary_image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)
    binary_image=judge_BinaryImage(binary_image)
    # 去掉外部白色边框，以免查找轮廓时仅找到外框
    offsetX = 3
    offsetY = 5
    offset_region = binary_image[offsetY:-offsetY, offsetX:-offsetX]
    return offset_region
#基于边缘特征的字符分割
def get_candidate_chars1(candidate_plate_image,offset_region):
    #working_region用于汉字模糊处理获取轮廓，offset_region用于分割
    working_region = np.copy(offset_region)

    # 仅将汉字字符所在区域模糊化，使得左右结构或上下结构的汉字不会被识别成多个不同的轮廓
    chinese_char_max_width = working_region.shape[1] // 8;		# 假设汉字最大宽度为整个车牌宽度的1/8
    chinese_char_region = working_region[:, 0:chinese_char_max_width]
    cv.GaussianBlur(chinese_char_region, (9, 9), 0, dst=chinese_char_region)            # 采用In-Place平滑处理

    # 在工作区中查找轮廓
    char_contours, _ = cv.findContours(working_region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # char_contours, _ = cv.findContours(working_region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(candidate_plate_image, char_contours, -1, (0, 0, 255))
    # cv.imshow("", candidate_plate_image)
    # cv.waitKey()

    #以下假设字符的最小宽度是车牌宽度的1/40（考虑到I这样的字符），高度是80%
    min_width = working_region.shape[1] // 40
    min_height = working_region.shape[0] * 7// 10
    valid_char_regions = []
    for i in np.arange(len(char_contours)):
        x, y, w, h = cv.boundingRect(char_contours[i])
        #字符高度和宽度必须满足条件
        if h >= min_height and w >= min_width and w<h:
            valid_char_regions.append((x, offset_region[y:y+h, x:x+w]))
    # 按照轮廓的x坐标从左到右排序
    sorted_regions = sorted(valid_char_regions, key=lambda region:region[0])
    # 包装到一个list中返回
    candidate_char_images = []
    for i in np.arange(len(sorted_regions)):
        candidate_char_images.append(sorted_regions[i][1])

    return candidate_char_images

def get_grey_binary_image2(candidate_plate_image):
    # 灰度化和二值化该区域
    gray_image = cv.cvtColor(candidate_plate_image, cv.COLOR_BGR2GRAY)
    ret, binary_image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)
    # 去掉外部白色边框，以免查找轮廓时仅找到外框
    offsetX = 3
    offsetY = 5
    offset_region = binary_image[offsetY:-offsetY, offsetX:-offsetX]
    return offset_region
#基于垂直投影的字符分割
def get_candidate_chars2(candidate_plate_image, offset_region):
    working_region = np.copy(offset_region)

    #cv.GaussianBlur(working_region, (11,11), 0, dst=working_region)  # 采用In-Place平滑处理

    height=working_region.shape[0]
    width=working_region.shape[1]

    max_white=0
    max_black=0

    white=[]
    black=[]
    for i in range(width):
        white_num=0
        black_num=0
        for j in range(height):
            if working_region[j][i] == 255:
                white_num+=1

            if working_region[j][i] ==0:
                black_num+=1

        max_white=max(max_white,white_num)
        max_black=max(max_black,black_num)
        white.append(white_num)
        black.append(black_num)

    #判断白底还是黑底
    wb=[]
    bw=[]
    max_wb=0#垂直方向最大值
    max_bw=0#水平方向最大值
    if max_white > max_black:#白底
        wb=black
        bw=white
        max_wb=max_black
        max_bw=max_white
    else:#黑底
        wb=white
        bw=black
        max_wb=max_white
        max_bw=max_black

    # plt.plot(wb)
    # plt.show()

    #以下假设字符的最小宽度是车牌宽度的1/40（考虑到I这样的字符），高度是80%
    min_width = working_region.shape[1] // 40
    min_height = working_region.shape[0] * 7// 10

    valid_char_regions = []
    index=1
    start=1
    end=2
    while index<width-2:
        index+=1
        if wb[index]>0.05*max_wb:
            start=index

            s=start
            e=s+1
            for i in range(s+1,width-1):
                if bw[i] > 0.95*max_bw:
                    e=i
                    break
            end = e
            index=end
            if end-start > 5:
                valid_char_regions.append((start, working_region[:,start:end]))

    # 按照轮廓的x坐标从左到右排序
    sorted_regions = sorted(valid_char_regions, key=lambda region:region[0])
    # 包装到一个list中返回
    candidate_char_images = []
    for i in np.arange(len(sorted_regions)):
        candidate_char_images.append(sorted_regions[i][1])

    return candidate_char_images

def edge_seg(image):
    color=judgecolor(image)
    offset_region = get_grey_binary_image1(image)
    #cv.imshow("", offset_region)
    cv.waitKey()
    candidate_chars = get_candidate_chars1(image, offset_region)
    candidate_chars = Square(224, candidate_chars)
    '''
        for char in candidate_chars:
        cv.imshow("", char)
        cv.waitKey()
    cv.destroyAllWindows()
    '''

    return candidate_chars, color

def proj_seg(image):
    judgecolor(image)
    offset_region = get_grey_binary_image2(image)
    # cv.imshow("", offset_region)
    # cv.waitKey()
    candidate_chars = get_candidate_chars2(image, offset_region)
    candidate_chars = Square(224, candidate_chars)
    '''
    for char in candidate_chars:
        cv.imshow("", char)
        cv.waitKey()
    cv.destroyAllWindows()
    '''

    return candidate_chars


'''
if __name__ == '__main__':
    #candidate_plate_image为修正后的图片

    #二值化
    offset_region = get_grey_binary_image(candidate_plate_image)

    #分割
    # candidate_chars = get_candidate_chars1(candidate_plate_image,offset_region)
    candidate_chars = get_candidate_chars2(candidate_plate_image, offset_region)
'''

'''
test_part2.py用于查看分割结果
所有用到的函数均在segmentation.py里

该部分的操作：将矫正后的图片先二值化，再去进行分割
里面有些函数也许三个部分可以共享，后面再整合

写了两种方法：
基于边缘特征的字符分割 和 基于垂直投影的字符分割
第一种方法在老师给的基础上做的改进
第二种看了一篇论文查资料写的
可以测试一下哪种好用哪个
如果确定用第二种我可以再改改函数，再简洁一点

基于垂直投影的字符分割也可以使用get_grey_binary_image1进行二值化

get_grey_binary_image1相比于get_grey_binary_image2多做了一步操作：
判断二值化后binary image是黑底白字还是白底黑字，若为白底黑字会反转为黑底白字

judgecolor函数用来判断车牌颜色（我也不知道有啥用），只能判断 蓝 黄 绿 三种颜色

Reverse函数是将二值图黑白颜色颠倒的函数

前后连接：
输入：  来自  第一部分    一张切割矫正好的车牌图片
输出：  传给  第三部分    一个图片数组，每张图片是一个切割好的字符 字符按顺序排放  
    candidate_chars[0]省份简称 candidate_chars[1]城市代码  后面是数字


因为没有测试过矫正的到的图片，不能完全保证性能，如果数据处理得到的图片达到理想预期的化应该问题不大

可能确定尺寸之后里面一些数值还要修改
'''
