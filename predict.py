import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from model import AlexNet_v1

def predict_chinese(model, char_image, class_indict):
    im_height = 224
    im_width = 224
    cv2.imwrite('333.jpg', char_image)
    # load image
    src = cv2.imread("333.jpg")
    img = cv2.resize(src, (im_width, im_height))
    # scaling pixel value to (0-1)
    img = np.array(img) / 255.
    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    # prediction
    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)
    #print(class_indict[str(predict_class)], result[predict_class])
    return class_indict[str(predict_class)]

def predict_others(model, char_image, class_indict):
    im_height = 224
    im_width = 224
    cv2.imwrite('333.jpg', char_image)
    # load image
    src = cv2.imread("333.jpg")
    img = cv2.resize(src, (im_width, im_height))

    # scaling pixel value to (0-1)
    img = np.array(img) / 255.

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    # prediction
    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)
    #print(class_indict[str(predict_class)], result[predict_class])
    return class_indict[str(predict_class)]

def pred(chars):
    res = []

    # create model
    model_c = AlexNet_v1(num_classes=31)
    model_c.load_weights("save_weights/myAlex_chinese.h5")

    model_e = AlexNet_v1(num_classes=34)
    model_e.load_weights("./save_weights/myAlex.h5")

    # read class_indict
    json_path_c = 'class_indices_chinese.json'
    assert os.path.exists(json_path_c), "file: '{}' dose not exist.".format(json_path_c)
    with open(json_path_c, "r") as f:
        class_indict_c = json.load(f)

    json_path_e = './class_indices.json'
    assert os.path.exists(json_path_e), "file: '{}' dose not exist.".format(json_path_e)
    with open(json_path_e, "r") as f:
        class_indict_e = json.load(f)


    c_f = predict_chinese(model_c, chars[0], class_indict_c)
    res.append(c_f)
    del(chars[0])

    cnt = True
    for char_img in chars:
        c_s = predict_others(model_e, char_img, class_indict_e)
        res.append(c_s)

    return res