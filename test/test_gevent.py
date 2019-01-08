import time
import gevent
from gevent import monkey, pool
monkey.patch_all()


def aa(x):
    num = 0
    for i in range(x):
        num += i
    print(num)
    time.sleep(4)


p = pool.Pool(4)


for i in range(1000, 1003):

    p.spawn(aa, i)
t1 = time.time()
gevent.joinall(p)
t2 = time.time()
print(t2 - t1)

for i in range(1000, 1003):

    aa(i)
t3 = time.time()
print(t3-t2)


