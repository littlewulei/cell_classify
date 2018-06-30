import cv2
import os

img = cv2.imread('1.jpeg')#your picture path
cv2.imshow('original',img)

for l in range((img.shape)[0]):
    for w in range((img.shape)[1]):
        if (img[l][w].sum()<100):
            img[l][w] = [210,209,200]
cv2.imshow('removw black',img)

hsv_img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)#HSV空间
h,s,v = cv2.split(hsv_img)
cv2.imshow('s',s)
ret,binary = cv2.threshold(s,90,255,cv2.THRESH_BINARY_INV)#二值化
img_medianBlur=cv2.medianBlur(binary,5) # 核越大，能去除的噪点越大
cv2.imshow('binary',binary)
cv2.imshow('medianBlur',img_medianBlur)
cv2.waitKey(0)
cv2.destroyAllWindows()
