test_img_path = []
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
'''
# 展示其中大熊猫图片
img1 = mpimg.imread(test_img_path[0])

plt.figure(figsize=(10, 10))
plt.imshow(img1)

plt.axis('off')
plt.show()
'''
with open('test1.txt', 'r') as f:
    test_img_path=[]
    for line in f:
        test_img_path.append(line.strip())
print(test_img_path)

import paddlehub as hub

module = hub.Module(name="resnet50_vd_animals")

import cv2
np_images =[cv2.imread(image_path) for image_path in test_img_path]

results = module.classification(images=np_images)

for result in results:
    print(result)