from src.gen_modle_cn4 import (crack_captcha_cnn, MAX_CAPTCHA, CHAR_SET_LEN, X,
                               keep_prob, vec2text, convert2gray)
import tensorflow as tf
import cv2
import numpy as np
from src.gen_captcha import gen_captcha_text_and_image


def crack_captcha(captcha_image):
    output = crack_captcha_cnn()
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # saver.restore(sess, tf.train.latest_checkpoint('./src/modle_4_cn4'))
        saver.restore(sess, tf.train.latest_checkpoint('./src/modle_4'))
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

        text = text_list[0].tolist()
        vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
        i = 0
        for n in text:
                vector[i*CHAR_SET_LEN + n] = 1
                i += 1
        return vec2text(vector)


def get_pic_text(image_):
    image_ = convert2gray(image_)
    image_ = image_.flatten() / 255
    predict_text_ = crack_captcha(image_)
    return predict_text_


def image2array():
    image = cv2.imread("")
    image = np.array(image)
    return image


if __name__ == '__main__':
    # text, image = gen_captcha_text_and_image(train=False)
    image = cv2.imread("6R86.jpg")
    # image = np.array(image)
    # print(image)
    # print(image.shape)
    # image = image2array()
    # image = convert2gray(image)
    # image = image.flatten() / 255
    # print(image)
    predict_text = get_pic_text(image)
    print("正确: {}  预测: {}".format("21976", predict_text))
