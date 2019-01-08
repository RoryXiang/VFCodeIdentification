from src.gen_moddle_6t import (crack_captcha_cnn_6, MAX_CAPTCHA_6, CHAR_SET_LEN, X_6, keep_prob_6,
                               vec2text, convert2gray)
import tensorflow as tf
import cv2
import numpy as np
import os

test_path = "./src/temp_pic_6"
# test_path = "./src/trainImage_6"
# modle_path = "./src/modle_6_cn4"
modle_path = "./src/modle_6"


def crack_captcha(captcha_image_list):
    output = crack_captcha_cnn_6()

    saver = tf.train.Saver()
    txt_list = []
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(modle_path))

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA_6, CHAR_SET_LEN]),
                            2)
        for image in captcha_image_list:
            print("_"*10)
            text_list = sess.run(predict,
                                 feed_dict={X_6: [image], keep_prob_6: 1})

            text = text_list[0].tolist()
            vector = np.zeros(MAX_CAPTCHA_6 * CHAR_SET_LEN)
            i = 0
            for n in text:
                vector[i * CHAR_SET_LEN + n] = 1
                i += 1
            one_vector = vec2text(vector)
            print("::::::", one_vector)
            txt_list.append(one_vector)
        return txt_list


def get_pic_text_(image_list_):
    image_list = []
    for image in image_list_:
        image_ = convert2gray(image)
        image_ = image_.flatten() / 255
        image_list.append(image_)
    print("^^^^^^^", len(image_list_), len(image_list))
    predict_text_ = crack_captcha(image_list)
    return predict_text_


def image2array():
    image = cv2.imread("1534155407.894559.jpg")
    image = np.array(image)
    return image


if __name__ == '__main__':
    # text, image = gen_captcha_text_and_image(train=False)
    # image = cv2.imread("34S7.jpg")
    # image = np.array(image)
    # print(image)
    # print(image.shape)
    # image = image2array()
    # image = convert2gray(image)
    # image = image.flatten() / 255
    # print(image)
    pic_list = os.listdir(test_path)
    text_list_ = []
    image_list_ = []
    for name in pic_list:
        name_ = name.split(".")[0]
        text_list_.append(name_)
        image_ = cv2.imread(test_path + "/" + name)
        image_ = np.array(image_)
        image_list_.append(image_)
    predict_text_list = get_pic_text_(image_list_)
    print("正确: {}  预测: {}".format(text_list_, predict_text_list))
    i = 0
    for index, name in enumerate(text_list_):
        if name != predict_text_list[index]:
            i += 1
            print(name, predict_text_list[index])
        else:
            print("------>", name)
        #     os.remove("./src/temp_pic_4/{}.jpg".format(name))
    print(len(text_list_), ":------:", i)