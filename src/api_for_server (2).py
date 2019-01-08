import numpy as np
import base64
import cv2
from src.gen_modle_cn4_ import (MAX_CAPTCHA_4, CHAR_SET_LEN, X_4, keep_prob_4,
                                vec2text, convert2gray)
from src.gen_moddle_6t import (MAX_CAPTCHA_6, X_6, keep_prob_6)


def handle_image_shape(image):
    image_ = convert2gray(image)
    image_ = image_.flatten() / 255
    return image_


def image2text_4t(sess, predict, image):
    image = handle_image_shape(image)
    text_list = sess.run(predict,
                         feed_dict={X_4: [image], keep_prob_4: 1})

    text = text_list[0].tolist()
    vector = np.zeros(MAX_CAPTCHA_4 * CHAR_SET_LEN)
    i = 0
    for n in text:
        vector[i * CHAR_SET_LEN + n] = 1
        i += 1
    one_vector = vec2text(vector)
    return one_vector


def image2text_6t(sess, predict, image):
    image = handle_image_shape(image)
    text_list = sess.run(predict,
                         feed_dict={X_6: [image], keep_prob_6: 1})

    text = text_list[0].tolist()
    vector = np.zeros(MAX_CAPTCHA_6 * CHAR_SET_LEN)
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