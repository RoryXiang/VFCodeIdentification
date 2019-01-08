import numpy as np
import base64
import cv2
from src.gen_modle_cn4 import (MAX_CAPTCHA, CHAR_SET_LEN, X, keep_prob,
                               vec2text, convert2gray)


def handle_image_shape(image):
    image_ = convert2gray(image)
    image_ = image_.flatten() / 255
    return image_


def image2text(sess, predict, image):
    image = handle_image_shape(image)
    text_list = sess.run(predict,
                         feed_dict={X: [image], keep_prob: 1})

    text = text_list[0].tolist()
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    i = 0
    for n in text:
        vector[i * CHAR_SET_LEN + n] = 1
        i += 1
    one_vector = vec2text(vector)
    return one_vector


def image_base64_array(image_str):
    image = base64.b64decode(image_str)
    image = np.fromstring(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image