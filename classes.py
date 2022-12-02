import os
import cv2
import numpy as np
import numpy as np

path="./DatasetSI/"
# classes = []

# for filename in os.listdir(path):
#     print(filename)
#     classes.append(filename)

# testing background removal
image = cv2.imread('./DatasetSI/anadearmas/ana_de_armas2.jpg')

# threshold on white
# lowerT = np.array([200,200,200])
# upperT = np.array([255,255,255])

# thresh = cv2.inRange(image,lowerT,upperT)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
# morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# mask = 255 - morph

# result = cv2.bitwise_and(image, image, mask=mask)

# fgbg2 = cv2.bgsegm.createBackgroundSubtractorGMG()
# backSub = cv2.createBackgroundSubtractorMOG2()
# fgmask2 = fgbg2.apply(image)
# mask2 = backSub.apply(image)
# cv2.imshow('MOG', mask2)

# myImage = cv2.GaussianBlur(image,(5,5),0)
# bins=np.array([0,51,102,153,204,255])
# myImage[:,:,:] = np.digitize(myImage[:,:,:],bins,right=True)*51

# myimage_grey = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)
# ret,background = cv2.threshold(myimage_grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

# ret,foreground = cv2.threshold(myimage_grey,0,255,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)  #Currently foreground is only a mask
# foreground = cv2.bitwise_and(image,image, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
 
#     # Combine the background and foreground to obtain our final image
# finalimage = background+foreground

# myimage_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# ret,baseline = cv2.threshold(myimage_grey,127,255,cv2.THRESH_TRUNC)

# ret,background = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY)

# ret,foreground = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY_INV)

# foreground = cv2.bitwise_and(image,image, mask=foreground)  # Update foreground with bitwise_and to extract real foreground

# # Convert black and white back into 3 channel greyscale
# background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

# # Combine the background and foreground to obtain our final image
# finalimage = background+foreground


myimage_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
#Take S and remove any value that is less than half
s = myimage_hsv[:,:,1]
s = np.where(s < 127, 0, 1) # Any value below 127 will be excluded

# We increase the brightness of the image and then mod by 255
v = (myimage_hsv[:,:,2] + 127) % 255
v = np.where(v > 127, 1, 0)  # Any value above 127 will be part of our mask

# Combine our two masks based on S and V into a single "Foreground"
foreground = np.where(s+v > 0, 1, 0).astype(np.uint8)  #Casting back into 8bit integer

background = np.where(foreground==0,255,0).astype(np.uint8) # Invert foreground to get background in uint8
background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)  # Convert background back into BGR space
foreground=cv2.bitwise_and(myimage_hsv,myimage_hsv,mask=foreground) # Apply our foreground map to original image
finalimage = background+foreground # Combine foreground and background

cv2.imshow('mi esposa bella', finalimage)
cv2.waitKey(0)