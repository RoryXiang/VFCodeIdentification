from src.gen_modle_cn4_ import (crack_captcha_cnn, MAX_CAPTCHA, CHAR_SET_LEN, X,
                               keep_prob, vec2text, convert2gray)
import tensorflow as tf
import cv2
import numpy as np
import os

# test_path = "./src/trainImage_4"
test_path = "./src/temp_pic_4"
modle_path = "./src/modle_4_cn4"
# modle_path = "./src/modle_4"


def crack_captcha(captcha_image_list):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    txt_list = []
    with tf.Session() as sess:
        # saver.restore(sess, tf.train.latest_checkpoint('./src/modle_4_cn4'))
        saver.restore(sess, tf.train.latest_checkpoint(modle_path))

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]),
                            2)
        for image in captcha_image_list:
            print("_"*10)
            text_list = sess.run(predict,
                                 feed_dict={X: [image], keep_prob: 1})

            text = text_list[0].tolist()
            vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
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
    pic_list = os.listdir(test_path)
    print(len(pic_list))
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
        # else:
        #     os.remove("./src/temp_pic_4/{}.jpg".format(name))
    print(len(text_list_), ":------:", i)