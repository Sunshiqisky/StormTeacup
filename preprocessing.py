import cv2
import glob

imagepath = glob.glob("../data/all/2/*.jpg")
temp = 0
for item in imagepath:
    temp += 1
    img = cv2.imread(item)
    img = img[:980,:1200,:]
    cv2.imwrite(str(temp) + '.jpg', img)
    print(temp)
