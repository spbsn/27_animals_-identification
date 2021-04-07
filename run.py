# encoding:utf-8

import requests
import base64
from PIL import Image

'''
动物识别
'''

request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/animal"
# 二进制方式打开图片文件
f = open('C:\\Users\spbsn\Desktop\zhu.jpg', 'rb')
img = base64.b64encode(f.read())

params = {"image":img}
access_token = '24.c6b637a0ec72d7c899670c2a3d618d4b.2592000.1620309422.282335-23941742'
request_url = request_url + "?access_token=" + access_token
headers = {'content-type': 'application/x-www-form-urlencoded'}
response = requests.post(request_url, data=params, headers=headers)
if response:
    for i in range(len(response.json()["result"])):
        print (response.json()["result"][i])
'''
im = Image.open('C:\\Users\spbsn\Desktop\\ben.jpg')
im.show()
'''