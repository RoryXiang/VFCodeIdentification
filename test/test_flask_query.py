import requests
import cv2
import numpy as np
import json
import base64

ip_port_url = "http://116.196.118.3:1314/getIp"
pic_4_url = "http://www.miibeian.gov.cn/getDetailVerifyCode"

headers = {
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        "Cookie": "__jsluid=80a1f7484fc710cb23d50d531638187f; JSESSIONID=Xrv"
                  "ZuLpysg1UiUyts4ufad6r2_BNM4TQwjnxEfp_2UySwDeBG11i!-73661681"
                  "3; Hm_lvt_d7682ab43891c68a00de46e9ce5b76aa=1532661015; Hm_l"
                  "pvt_d7682ab43891c68a00de46e9ce5b76aa=1532661015",
        "Host": "www.miibeian.gov.cn",
        "Referer": "http://www.miibeian.gov.cn/icp/publish/query/icpMemoInfo"
                   "_showPage.action",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHT"
                      "ML, like Gecko) Chrome/67.0.3396.99 Safari/537.36",
        }


def get_text():
    ip_port = requests.get(ip_port_url).content.decode()
    proxies = {"http": "http://{}".format(ip_port)}
    res_statu = 0
    while not res_statu:
        try:
            pic_res = requests.get(
                url=pic_4_url,
                proxies=proxies,
                headers=headers,
                timeout=1
            )
            res_statu = pic_res.status_code
        except:
            pass
    pic_res = pic_res.content
    pic_res = base64.b64encode(pic_res)
    print(type(pic_res.decode()))
    print(pic_res.decode())
    text = requests.post(
        url="http://0.0.0.0:5000/getText",
        json={"image": pic_res.decode()}
    )
    print(text.json())


if __name__ == '__main__':
    get_text()
