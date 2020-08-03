# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 10:53:45 2020

these codes are adapted from MIT Treepedia project 
https://github.com/mittrees/Treepedia_Public

VegetationClassification function from GreenView_Calculate.py is modified to 
calculate green percentage from aerial photos like google map or so, and to 
save itermedia images into jpeg files. Furthermore, it is (hopefully) use to 
recoginise trees from grasses which shows higher value in green band.

@author: Mona
"""

def graythresh(array,level):
    '''array: is the numpy array waiting for processing
    return thresh: is the result got by OTSU algorithm
    if the threshold is less than level, then set the level as the threshold
    by Xiaojiang Li
    '''
    
    import numpy as np
    
    maxVal = np.max(array)
    minVal = np.min(array)
    
#   if the inputImage is a float of double dataset then we transform the data 
#   in to byte and range from [0 255]
    if maxVal <= 1:
        array = array*255
        # print "New max value is %s" %(np.max(array))
    elif maxVal >= 256:
        array = np.int((array - minVal)/(maxVal - minVal))
        # print "New min value is %s" %(np.min(array))
    
    # turn the negative to natural number
    negIdx = np.where(array < 0)
    array[negIdx] = 0
    
    # calculate the hist of 'array'
    dims = np.shape(array)
    hist = np.histogram(array,range(257))
    P_hist = hist[0]*1.0/np.sum(hist[0])
    
    omega = P_hist.cumsum()
    
    temp = np.arange(256)
    mu = P_hist*(temp+1)
    mu = mu.cumsum()
    
    n = len(mu)
    mu_t = mu[n-1]
    
    sigma_b_squared = (mu_t*omega - mu)**2/(omega*(1-omega))
    
    # try to found if all sigma_b squrered are NaN or Infinity
    indInf = np.where(sigma_b_squared == np.inf)
    
    CIN = 0
    if len(indInf[0])>0:
        CIN = len(indInf[0])
    
    maxval = np.max(sigma_b_squared)
    
    IsAllInf = CIN == 256
    if IsAllInf !=1:
        index = np.where(sigma_b_squared==maxval)
        idx = np.mean(index)
        threshold = (idx - 1)/255.0
    else:
        threshold = level
    
    if np.isnan(threshold):
        threshold = level
    
    return threshold

def VegetationClassification(ImgFileName): #, panoID, heading):
    '''
    This function is used to classify the green vegetation from GSV image,
    This is based on object based and otsu automatically thresholding method
    The season of GSV images were also considered in this function
        Img: the numpy array image, eg. Img = np.array(Image.open(StringIO(response.content)))
        return the percentage of the green vegetation pixels in the GSV image
    
    By Xiaojiang Li
    '''
    
    import pymeanshift as pms
    import numpy as np
    from PIL import Image
    
    # these lines are added to setup output file name and folder
    outputPath = 'output/' + ImgFileName.split('\\')[1]

    Img = np.array(Image.open(ImgFileName))

    # this line is added to save origin image into ouptut folder
    Image.fromarray(Img, 'RGB').save(outputPath + '_origin.jpg')
    
    # use the meanshift segmentation algorithm to segment the original GSV image
    (segmented_image, labels_image, number_regions) = pms.segment(Img,spatial_radius=6,
                                                     range_radius=7, min_density=40)

    # this line is added to save segmented image into ouptut folder
    Image.fromarray(segmented_image, 'RGB').save(outputPath + '_seg.jpg')

    I = segmented_image/255.0
    
    red = I[:,:,0]
    green = I[:,:,1]
    blue = I[:,:,2]
    
    # calculate the difference between green band with other two bands
    green_red_Diff = green - red
    green_blue_Diff = green - blue
    
    ExG = green_red_Diff + green_blue_Diff
    diffImg = green_red_Diff * green_blue_Diff
    
    redThreImgU = red < 0.6
    greenThreImgU = green < 0.9
    blueThreImgU = blue < 0.6
    
    shadowRedU = red < 0.3
    shadowGreenU = green < 0.3
    shadowBlueU = blue < 0.3
    del red, blue, I #green, I
    
    greenImg1 = redThreImgU * blueThreImgU * greenThreImgU
    greenImgShadow1 = shadowRedU * shadowGreenU * shadowBlueU
    del redThreImgU, greenThreImgU, blueThreImgU
    del shadowRedU, shadowGreenU, shadowBlueU
    
    greenImg3 = diffImg > 0.0
    greenImg4 = green_red_Diff > 0
    threshold = graythresh(ExG, 0.1)
    
    if threshold > 0.1:
        threshold = 0.1
    elif threshold < 0.05:
        threshold = 0.05
    
    greenImg2 = ExG > threshold
    greenImgShadow2 = ExG > 0.05
    greenImg = greenImg1 * greenImg2 + greenImgShadow2 * greenImgShadow1
    del ExG,green_blue_Diff,green_red_Diff
    del greenImgShadow1,greenImgShadow2

    # this line is added to save green-classfied image into ouptut folder
    Image.fromarray(greenImg).save(outputPath + '_greenImg.jpg')
    
    # calculate the percentage of the green vegetation
    greenPxlNum = len(np.where(greenImg != 0)[0])
    # this line is modified to calculate greenPercent base on image size instead of 400x400 resolution
    greenPercent = greenPxlNum/(greenImg.shape[0]*greenImg.shape[1])*100
    del greenImg1,greenImg2
    del greenImg3,greenImg4
    
    # these lines are added to calculate tree coverage (not scientifically assessed)
    greenImg5 = green < (np.max(green) - 3.8 * np.std(green))
    greenTREE = greenImg * greenImg5
    
    Image.fromarray(greenTREE).save(outputPath + '_greenTREE.jpg')

    greenTreePxlNum = len(np.where(greenTREE != 0)[0])
    greenTreePercent = greenTreePxlNum/(greenTREE.shape[0]*greenTREE.shape[1])*100
    del greenImg5,green
    
    return [greenPercent, greenTreePercent]

def main():
    import glob
    fileList = glob.glob('aerial_photo_trial/*.jpg')
    # fileList = os.listdir('aerial_photo_trial')
    
    for fileName in fileList:
        print('Doing ' + fileName + '...')    
        [thisGP, thisTreeGP] = VegetationClassification(fileName)
        print('GreenPercent of ' + fileName + ' is ' + str(thisGP))
        print('TreeGreenPercent of ' + fileName + ' is ' + str(thisTreeGP))

    