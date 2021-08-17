import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import *
import glob
from PIL import *
import math


def getMeanError(gt ,joints):
    """
    get average error over all joints, averaged over sequence
    :return: mean error
    """
   return numpy.nanmean(numpy.sqrt(numpy.square(gt - joints).sum(axis=2)), axis=1).mean()


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
  imageA= 'real_A' +imageN[6:]
  fullA=  'results\\handpose_5_new_test\\test_latest\\001\\'+ imageA
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
  p.write(str(centroid_Ind_depth) +',')
  
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
  p.write(str(centroid_Pnk_depth) +'\n')
  
    
    
  # to read and extract ground truth 2D coordinate

  image_labeled_org = 'real' + imageN[4:]
  fullB = 'results\\handpose_5_new_test\\test_latest\\001\\'+ image_labeled_org
  image_labeled = cv2.imread(fullB)
  Converted_image_labeled = cv2.cvtColor(image_labeled, cv2.COLOR_BGR2RGB)
  plt.imshow(Converted_image_labeled)
  hsv_Converted_image_labeled = cv2.cvtColor(Converted_image_labeled, cv2.COLOR_RGB2HSV)
  a.write(image_labeled_org +',')

  #Thumb
  light_red = ( 0, 51, 90)
  dark_red= (5,255,255)
  mask = cv2.inRange(hsv_Converted_image_labeled,light_red,dark_red)
  plt.figure(),plt.imshow(mask, cmap="gray")
  centroid_Thb = extract_joint(mask)
  print("Thumb",centroid_Thb)
  a.write(str(centroid_Thb[0]) +','+str(centroid_Thb[1]) +',')
  centroid_Thb_depth = testA [math.floor(centroid_Thb[0]),math.floor(centroid_Thb[1])]
  a.write(str(centroid_Thb_depth) +',')

  #Index
  light_cyan = (90, 124, 198)
  dark_cyan = (103, 255, 255)
  mask = cv2.inRange(hsv_Converted_image_labeled,light_cyan,dark_cyan)
  plt.figure(),plt.imshow(mask, cmap="gray")
  centroid_Ind = extract_joint(mask)
  print("Index",centroid_Ind)
  plt.figure()
  a.write(str(centroid_Ind[0]) +','+str(centroid_Ind[1]) +',')
  centroid_Ind_depth = testA [math.floor(centroid_Ind[0]),math.floor(centroid_Ind[1])]
  a.write(str(centroid_Ind_depth) +',')

  #Middle
  light_yellow = (9, 72, 157)
  dark_yellow  = (80, 255, 255)
  mask = cv2.inRange(hsv_Converted_image_labeled,light_yellow,dark_yellow)
  plt.figure(),plt.imshow(mask, cmap="gray")
  centroid_Mdl = extract_joint(mask)
  print("Middle",centroid_Mdl)
  a.write(str(centroid_Mdl[0]) +','+str(centroid_Mdl[1]) +',')
  centroid_Mdl_depth = testA [math.floor(centroid_Mdl[0]),math.floor(centroid_Mdl[1])]
  a.write(str(centroid_Mdl_depth) +',')

  #Ring
  light_green = ( 40, 98, 39)
  dark_green= (80,255,255)
  mask = cv2.inRange(hsv_Converted_image_labeled,light_green,dark_green)
  plt.figure(),plt.imshow(mask, cmap="gray")
  centroid_Rng = extract_joint(mask)
  print("Ring",centroid_Rng)
  a.write(str(centroid_Rng[0]) +','+str(centroid_Rng[1]) +',')
  centroid_Rng_depth = testA [math.floor(centroid_Rng[0]),math.floor(centroid_Rng[1])]
  a.write(str(centroid_Rng_depth) +',')

  #Pinky
  light_blue = ( 99, 250, 0)
  dark_blue= (179,255 ,255)
  mask = cv2.inRange(hsv_Converted_image_labeled,light_blue,dark_blue)
  plt.figure(),plt.imshow(mask, cmap="gray")
  centroid_Pnk = extract_joint(mask)
  print("Pinky",centroid_Pnk)
  a.write(str(centroid_Pnk[0]) +','+str(centroid_Pnk[1]) +',')
  centroid_Pnk_depth = testA [math.floor(centroid_Pnk[0]),math.floor(centroid_Pnk[1])]
  a.write(str(centroid_Pnk_depth) +'\n')

p.close()
a.close()


gt = np.loadtxt("actual_handpose_5_new_test.txt", dtype=float, delimiter=",",usecols=range(1,16))
name_actual = np.loadtxt("actual_handpose_5_new_test.txt", dtype=str, delimiter=",",usecols=range(0,1))
gt= gt.reshape(-1,5,3)

pr = np.loadtxt("predicted_handpose_5_new_test.txt", dtype=float, delimiter=",",usecols=range(1,16))
name_predicted = np.loadtxt("predicted_handpose_5_new_test.txt", dtype=str, delimiter=",",usecols=range(0,1))
pr= pr.reshape(-1,5,3)

ME = getMeanError(gt[:,:,:2],pr[:,:,:2])
print("2D ME on example video of 30 frames (in pixels):", ME)
