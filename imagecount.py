# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:24:56 2020

@author: xzk
"""

import cv2

def get_contour(filename):
    img = cv2.imread(filename)

    ## Remove ruler(x=1235, y=980)
    img1 = img.copy()
    img[-50:,-125:] = 0

    ## Convert to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## Convert to binary
    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    ## Find contour
    _, contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    return img1, contours


def main():
    # input image path
    filename = "1.jpg"
    output = 'test.jpg'
    img1, contours = get_contour(filename)

    # output number of cells and save image with contours
    print(len(contours))

    cv2.drawContours(img1, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(output, img1)

if __name__ == '__main__':
    main()
