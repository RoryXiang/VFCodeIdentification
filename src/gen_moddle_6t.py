from src.gen_captcha_6t import gen_captcha_text_and_image
from src.gen_captcha_6t import number
from src.gen_captcha_6t import ALPHABET

import numpy as np
import tensorflow as tf

# text, image = gen_captcha_text_and_image()
# print("验证码图像channel:", image.shape)  # (60, 160, 3)
# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 200
MAX_CAPTCHA_6 = 6  # len(text)
# print("验证码文本最长字符数", MAX_CAPTCHA, text)  # 验证码最长6字符; 我全部固定为6,可以不固定.
# 如果验证码长度小于6，用'_'补齐


# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


"""
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，
下补3行，左补2行，右补2行
"""

# 文本转向量
char_set = number + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(char_set)


def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA_6:
        raise ValueError('验证码最长6个字符')

    vector = np.zeros(MAX_CAPTCHA_6 * CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        if idx > 222:
            print(text)
        vector[idx] = 1
    return vector


# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


"""
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有
vec = text2vec("F5Sd")
text = vec2text(vec)
print(text)  # F5Sd
vec = text2vec("SFd5")
text = vec2text(vec)
print(text)  # SFd5
"""


# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA_6 * CHAR_SET_LEN])

    # 有时生成图像大小不是(60, 200, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (60, 200, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i,
        :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


####################################################################

X_6 = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y_6 = tf.placeholder(tf.float32, [None, MAX_CAPTCHA_6 * CHAR_SET_LEN])
keep_prob_6 = tf.placeholder(tf.float32)  # dropout


# 定义CNN
def crack_captcha_cnn_6(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X_6, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 4 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([5, 5, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))

    print("w_c1: ", w_c1.shape)
    print(X_6.shape)
    print("@@@@@@@@@@@@")
    print("x: ", x.shape)
    print("b_c1", b_c1.shape)
    conv1 = tf.nn.relu(tf.nn.bias_add(
        tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob_6)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([5, 5, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(
        tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob_6)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([5, 5, 64, 128]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv3 = tf.nn.relu(tf.nn.bias_add(
        tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob_6)

    w_c4 = tf.Variable(w_alpha * tf.random_normal([5, 5, 128, 256]))
    b_c4 = tf.Variable(b_alpha * tf.random_normal([256]))
    conv4 = tf.nn.relu(tf.nn.bias_add(
        tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4))
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')
    conv4 = tf.nn.dropout(conv4, keep_prob_6)

    w_c5 = tf.Variable(w_alpha * tf.random_normal([5, 5, 256, 512]))
    b_c5 = tf.Variable(b_alpha * tf.random_normal([512]))
    conv5 = tf.nn.relu(tf.nn.bias_add(
        tf.nn.conv2d(conv4, w_c5, strides=[1, 1, 1, 1], padding='SAME'), b_c5))
    conv5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')
    conv5 = tf.nn.dropout(conv5, keep_prob_6)

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([2 * 7 * 512, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv5, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob_6)

    w_out = tf.Variable(
        w_alpha * tf.random_normal([1024, MAX_CAPTCHA_6 * CHAR_SET_LEN]))
    b_out = tf.Variable(
        b_alpha * tf.random_normal([MAX_CAPTCHA_6 * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return out


# 训练
def train_crack_captcha_cnn():
    output = crack_captcha_cnn_6()
    # loss
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y_6))
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA_6, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y_6, [-1, MAX_CAPTCHA_6, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(100)
            _, loss_ = sess.run([optimizer, loss],
                                feed_dict={X_6: batch_x, Y_6: batch_y,
                                           keep_prob_6: 0.4})
            print("step: {}, loss: {}".format(step, loss_))

            # 每100 step计算一次准确率
            if step % 50 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy,
                               feed_dict={X_6: batch_x_test, Y_6: batch_y_test,
                                          keep_prob_6: 1.})
                print("iter step: {}, accecery is: {}".format(step, acc))
                # 如果准确率大于98%,保存模型,完成训练
                if acc > 0.98:
                    saver.save(sess, "./src/modle_6_cn4/crack_capcha.model",
                               global_step=step)
                    break

            step += 1


if __name__ == '__main__':

    train_crack_captcha_cnn()