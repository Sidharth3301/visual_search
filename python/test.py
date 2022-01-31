import cv2
from skimage import io
import numpy as np

# im = io.imread("sampleimg/targetgray004.jpg", plugin="pil")
# print(im.shape)
file = "croppednaturaldesign_img.txt"
f = open(file, mode="r", newline="\n")
l = f.readlines()
print(l.strip())
