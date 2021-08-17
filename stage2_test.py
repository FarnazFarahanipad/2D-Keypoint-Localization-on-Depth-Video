import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import *
import glob
from PIL import *
import math

def loadDepthMap(filename):
    """
    Read a depth-map
    :param filename: file name to load
    :return: image data of depth image
    """
    with open(filename) as f:
        img = Image.open(filename)
        # top 8 bits of depth are packed into green channel and lower 8 bits into blue
        assert len(img.getbands()) == 3
        r, g, b = img.split()
        r = np.asarray(r,np.int32)
        g = np.asarray(g,np.int32)
        b = np.asarray(b,np.int32)
        dpt = np.bitwise_or(np.left_shift(g,8),b)
        imgdata = np.asarray(dpt,np.float32)

    return imgdata

def extract_joint(mask):

  img = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
  # print(stats)
  plt.scatter(centroids[1,0],centroids[1,1],marker ='+',color ='r')
  return (centroids[1])

p= open("predicted_handpose_5_new_test.txt", "w+")
a= open("actual_handpose_5_new_test.txt", "w+")

for image in glob.glob('results\\handpose_5_new_test\\test_latest\\001\\fake_B_*.jpg'):
  test = cv2.imread(image)
  imageN = os.path.basename (image)
  imageB=imageN[:4]+'_real_B.jpg'
  fullA=  'results\\handpose_5_new_test\\test_latest\\001\\'+ imageB
  testA = cv2.imread(fullA,  cv2.IMREAD_ANYDEPTH)
  ConvertedTest = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
  hsv_ConvertedTest = cv2.cvtColor(ConvertedTest, cv2.COLOR_RGB2HSV)
  p.write(imageN +',')
  
  #Thumb
  light_red = ( 0, 51, 90)
  dark_red= (5,255,255)
  mask = cv2.inRange(hsv_ConvertedTest,light_red,dark_red)
  plt.figure(),plt.imshow(mask, cmap="gray")
  centroid_Thb = extract_joint(mask)
  print("Thumb",centroid_Thb)
  p.write(str(centroid_Thb[0]) +','+str(centroid_Thb[1]) +',')
  centroid_Thb_depth = testA [math.floor(centroid_Thb[0]),math.floor(centroid_Thb[1])]
  p.write(str(centroid_Thb_depth) +',')
  
  #Index
  light_cyan = (90, 124, 198)
  dark_cyan = (103, 255, 255)
  mask = cv2.inRange(hsv_ConvertedTest,light_cyan,dark_cyan)
  plt.figure(),plt.imshow(mask, cmap="gray")
  centroid_Ind = extract_joint(mask)
  print("Index",centroid_Ind)
  p.write(str(centroid_Ind[0]) +','+str(centroid_Ind[1]) +',')
  centroid_Ind_depth = testA [math.floor(centroid_Ind[0]),math.floor(centroid_Ind[1])]
  p.write(str(centroid_Ind_depth) +'\n')
  
  #Middle
  light_yellow = (9, 72, 157)
  dark_yellow  = (80, 255, 255)
  mask = cv2.inRange(hsv_ConvertedTest,light_yellow,dark_yellow)
  plt.figure(),plt.imshow(mask, cmap="gray")
  centroid_Mdl = extract_joint(mask)
  print("Middle",centroid_Mdl)
  p.write(str(centroid_Mdl[0]) +','+str(centroid_Mdl[1]) +',')
  centroid_Mdl_depth = testA [math.floor(centroid_Mdl[0]),math.floor(centroid_Mdl[1])]
  p.write(str(centroid_Mdl_depth) +',')
  
  #Ring 
  light_green = ( 40, 98, 39)
  dark_green= (80,255,255)
  mask = cv2.inRange(hsv_ConvertedTest,light_green,dark_green)
  plt.figure(),plt.imshow(mask, cmap="gray")
  centroid_Rng = extract_joint(mask)
  print("Ring",centroid_Rng)
  p.write(str(centroid_Rng[0]) +','+str(centroid_Rng[1]) +',')
  centroid_Rng_depth = testA [math.floor(centroid_Rng[0]),math.floor(centroid_Rng[1])]
  p.write(str(centroid_Rng_depth) +',')
  
  #Pinky 
  light_blue = ( 99, 250, 0)
  dark_blue= (179,255 ,255)
  mask = cv2.inRange(hsv_ConvertedTest,light_blue,dark_blue)
  plt.figure(),plt.imshow(mask, cmap="gray")
  centroid_Pnk = extract_joint(mask)
  print("Pinky",centroid_Pnk)
  p.write(str(centroid_Pnk[0]) +','+str(centroid_Pnk[1]) +',')
  centroid_Pnk_depth = testA [math.floor(centroid_Pnk[0]),math.floor(centroid_Pnk[1])]
  p.write(str(centroid_Pnk_depth) +',')
  
  
