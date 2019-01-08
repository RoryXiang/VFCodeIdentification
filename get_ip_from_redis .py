from redis import StrictRedis
import time
import requests
import random

def get_proxies():

    client = StrictRedis(host="172.16.63.61", port="6379",password="redis123456", db=10)
    ip_port = None
    while not ip_port:
        try:
            keys = client.keys()
            ip_port = client.get(random.choice(keys).decode())
            ip_port = ip_port.decode()
        except:
            pass
    return ip_port


def get_proxies_():
    ip_port = None
    while not ip_port:
        try:
            ip_port = requests.get("http://172.16.60.50:1314/getIp")
            # ip_port = requests.get("http://116.196.118.3:1314/getIp")
            ip_port = ip_port.content.decode()
        except:
            pass
    print("ip_port:", ip_port)
    return ip_port


if __name__ == '__main__':
    proxies = get_proxies()
    print(proxies)
