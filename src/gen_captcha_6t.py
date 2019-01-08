# from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
import random
import os
import cv2

# 验证码中的字符, 就不用汉字了
number = ['0','1','2','3','4','5','6','7','8','9']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
# 验证码一般都无视大小写；验证码长度6个字符

train_path = "./src/trainImage_6"
test_path = "./src/temp_pic_6"


def random_captcha_text(char_set=number + ALPHABET, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

# 生成字符对应的验证码


def gen_captcha_text_and_image(train=True):
    if train:
        path_ = train_path
    else:
        path_ = test_path
    pic_list = os.listdir(path_)
    name = random.choice(pic_list)
    captcha_text = name.split(".")[0]
    captcha_image = cv2.imread(path_ + "/" + name)
    # print(captcha_image)
    captcha_image = np.array(captcha_image)
    # print(captcha_image)
    return captcha_text, captcha_image


if __name__ == '__main__':
    pass