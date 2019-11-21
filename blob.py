# Sailent Regions
# https://github.com/NLeSC/SalientDetector-python/blob/master/Notebooks/DetectorExample.ipynb
# python -m pip install pip==9.0.3
# pip install salientregions 
# change back: python -m pip install --upgrade pip
# import salientregions as sr

import numpy as np
import cv2 as cv2


# Read in the image in grayscale
oimg = cv2.imread('images/test1.png')
img = cv2.imread('images/test1.png', 0)

# create SURF object
surf = cv2.xfeatures2d.SURF_create(400)

# Find keypoints and decriptors
kp, des = surf.detectAndCompute(img,None)
print( len(kp) )

# Increase Hessian Threshold to reduce number of keypoints
surf.setHessianThreshold(5000)

# Dont detect orientation
surf.setUpright(True)

# Find  NEW keypoints and decriptors
kp, des = surf.detectAndCompute(img,None)
print( len(kp) )

img2 = cv2.drawKeypoints(oimg,kp,None,(255,0,0),4)

cv2.imwrite('images/test1_surf.png', img2)


cv2.imshow('test',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save Cropped Images
i = 0
for keyPoint in kp:
    
    x = int(keyPoint.pt[0])
    y = int(keyPoint.pt[1])
    s = keyPoint.size
    r = int(s/2)
    cv2.circle(img,(x,y), r, (0,0,255), -1)
#    left_x = x - r
#    right_x = x + r
#    top_y = y + r
#    bottom_y = y - r
    cropped_img = oimg[y-r:y+r, x-r:x+r]
    resized_img = cv2.resize(cropped_img, (28, 28), interpolation = cv2.INTER_AREA)
    cv2.imwrite('images/cropped_images/' + str(i) + '.png', resized_img)
    i += 1

# Resize Cropped Images

    
   
cv2.imshow('test',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('images/test1_surf_blob.png', img)


image = cv2.imread('images/test1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
edged = cv2.Canny(blurred, 50, 150)

cv2.imwrite('images/test1_edge.png', edged)

cv2.imshow('detected circles',edged)
cv2.waitKey(0)
cv2.destroyAllWindows()





