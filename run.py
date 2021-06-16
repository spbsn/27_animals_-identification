# encoding:utf-8

import requests
import base64
from PIL import Image

'''
动物识别
'''
# encoding:utf-8
import requests

# client_id 为官网获取的AK， client_secret 为官网获取的SK
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=pzq2GGggP1WoiRb58VGINy0b&client_secret=ggacoyc2PembbjYNoHMRfENXBpqcGPyS'
response = requests.get(host)
if response:
    print(response.json())

request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/animal"
# 二进制方式打开图片文件
f = open('hhh.jpg', 'rb')
img = base64.b64encode(f.read())

params = {"image":img}
access_token = '24.428ddb07b0e16f05f5cb75b089910e47.2592000.1626001363.282335-23941742'
request_url = request_url + "?access_token=" + access_token
headers = {'content-type': 'application/x-www-form-urlencoded'}
response = requests.post(request_url, data=params, headers=headers)

if response:
    for i in range(len(response.json()['result'])):
        print(response.json()["result"][i])
'''
if response:
    print(response.json())
'''
'''
im = Image.open('C:\\Users\spbsn\Desktop\\ben.jpg')
im.show()
'''