'''
  File name: getFeatures.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detect features within each detected bounding box
    - Input img: the first frame (in the grayscale) of video
    - Input bbox: the four corners of bounding boxes
    - Output x: the x coordinates of features
    - Output y: the y coordinates of features
'''

def getFeatures(img, bbox):
    from skimage.feature import corner_shi_tomasi, corner_peaks
    import numpy as np
    N = 100
    x = np.zeros((N,len(bbox)))
    y = np.zeros((N,len(bbox)))
  #TODO: Your code here
  for i in range(len(bbox)):
      face = img[int(bbox[i][1][0]):int(bbox[i][3][0]),int(bbox[i][1][1]):int(bbox[i][3][1])]
      
      points = corner_peaks(corner_shi_tomasi(face),min_distance = 1)
      for j in range(len(points)):
          x[j,i] = points[j,0]
          y[j,i] = points[j,1]
  return x, y