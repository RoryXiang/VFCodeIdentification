from src.gen_model import (crack_captcha_cnn, MAX_CAPTCHA, CHAR_SET_LEN, X,
                           keep_prob, vec2text, convert2gray)
import numpy as np
from src.gen_captcha import gen_captcha_text_and_image
import tensorflow as tf


def crack_captcha(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./modle_4'))

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

        text = text_list[0].tolist()
        vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
        i = 0
        for n in text:
                vector[i*CHAR_SET_LEN + n] = 1
                i += 1
        return vec2text(vector)


text, image = gen_captcha_text_and_image(train=False)
image = convert2gray(image)
image = image.flatten() / 255
predict_text = crack_captcha(image)
print("正确: {}  预测: {}".format(text, predict_text))

