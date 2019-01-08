import json
from flask import Flask, request
import tensorflow as tf
from src.api_for_server import image2text_6t, image_base64_array
from src.gen_moddle_6t import crack_captcha_cnn_6, MAX_CAPTCHA_6, CHAR_SET_LEN

app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL="redis://localhost:6379",
    result_backend='redis://locbalhost:6379')


output_6 = crack_captcha_cnn_6()
saver_6 = tf.train.Saver()
sess_6 = tf.Session()
saver_6.restore(sess_6, tf.train.latest_checkpoint('./src/modle_6'))
predict_6 = tf.argmax(tf.reshape(output_6, [-1, MAX_CAPTCHA_6, CHAR_SET_LEN]),
                      2)


@app.route("/get6Text", methods=["POST"])
def get_6_text():
    query = request.get_json()
    image_str = query["image"]
    image = image_base64_array(image_str)
    if image is None:
        return json.dumps("N")
    text = image2text_6t(sess_6, predict_6, image)
    return json.dumps(text)


@app.route("/test", methods=["GET"])
def aa():
    return "hello"


if __name__ == '__main__':
    print("flask sever now is start!!!!!")
    app.run(threaded=True, host='0.0.0.0', port=5006)
