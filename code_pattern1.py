# from typing import Pattern
# from cv2 import imwrite
# from numpy.core.records import array
# from numpy.lib.type_check import imag
import cv2
# import matplotlib
import numpy as np
from skimage.filters import threshold_sauvola
from skimage import img_as_ubyte
# import glob
# import array as arr
import imutils
# import random as rng
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

#Thresholding this method work with grayscale return with bw
def thresh_sauvola(image,ksize) :
    thresh_sauvola = threshold_sauvola(image, window_size=ksize)
    binary_sauvola = image > thresh_sauvola
    image = cv2.bitwise_not(img_as_ubyte(binary_sauvola))
    return image

#This method work with graysclale return contour
def find_border(image):
    img = image
    gray = cv2.GaussianBlur(img, (7, 7), 0)
    gray= cv2.medianBlur(gray, 3)   #to remove salt and paper noise
    #to binary
    thresh = thresh_sauvola(gray,73)  #to detect white objects

    kernel = np.ones((2,2),np.uint8)
    #to get outer boundery only     
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
    #to strength week pixels
    thresh = cv2.dilate(thresh,kernel,iterations = 5)
    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key = cv2.contourArea)

    return c

#This method work with grayscale and contour
def delete_border(image,c):

    # Draw contours
    _,noborder = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cv2.drawContours(noborder, [c], -1 ,(0,0,0),80)
    return noborder

#This method work with color image return bw
def make_pattern(image,c, dila, ero):
    
    lower = np.array([0, 0, 0], dtype = "uint8")
    #upper = np.array([60, 60, 60], dtype = "uint8") #ori
    upper = np.array([70, 70, 70], dtype = "uint8")

# Test-2 color segmentation : up upper bound to 85
    #upper = np.array([97, 93, 82], dtype = "uint8")

    mask = cv2.inRange(image, lower, upper)
    delete = delete_border(mask,c)
    

# Test-1 edge detection : ori. canny    
    #edges = cv2.Canny(delete, 0, 122)

# Test-1.1 edge detection : combination sobel
    
    sobel_x = cv2.Sobel(delete, -1, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(delete, -1, 0, 1, ksize=5)
    edges = cv2.addWeighted(sobel_x, 1, sobel_y, 1, 0)
    
# Test-1.2 edge detection : normal sobel
    #edges = cv2.Sobel(delete, -1, 1, 1, ksize=5)

# Test-1.3 edge detection : laplacian
    #edges = cv2.Laplacian(delete,-1)


    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=40)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(delete, (x1, y1), (x2, y2), (255, 255, 255), 15)


    kernel = np.ones((7,7),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    kernel3 = np.ones((3,3),np.uint8)

    #---- ori: dila = ero = 10, new: dila =10, ero = 13 ----#
    dilation = cv2.dilate(delete,kernel,iterations = dila)
    erosion = cv2.erode(dilation,kernel,iterations = ero)

    return erosion

#This method work with color image return bw
def recheck_pattern(image,c,dila,ero):

    lower = np.array([0, 0, 0], dtype = "uint8")
    #upper = np.array([60, 60, 60], dtype = "uint8") #ori
    upper = np.array([70, 70, 70], dtype = "uint8")
    mask = cv2.inRange(image, lower, upper)
    mask = cv2.medianBlur(mask,3)
    delete = delete_border(mask,c)

    kernel = np.ones((7,7),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(delete, cv2.MORPH_OPEN, (5,5))

    #---- ori: dila = ero = 10, new: dila = 10, ero = 16 ----#
    dilation = cv2.dilate(mask,kernel,iterations = dila)
    erosion = cv2.erode(dilation,kernel2,iterations = ero)
    return erosion


def draw_error(image,result):
    img = image
    res = result
    # src_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # res = cv2.blur(res, (5,5))
    threshold = 0 # initial threshold
    # Detect edges using Canny
    canny_output = cv2.Canny(res, threshold, threshold * 2)
    cnts = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # list = []
    for cntr in cnts:
        x,y,w,h = cv2.boundingRect(cntr)
       
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 5)
        cv2.drawContours(img, [cntr], -1 ,(0,0,255),7)
        # list.append(area)

    return img

def draw_error2(image,result):
    img = image
    res = result
    threshold = 0
    canny_output = cv2.Canny(res, threshold, threshold * 2)
    cnts = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for cntr in cnts:
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.drawContours(img, [cntr], -1 ,(0,255,0),7)

    return img

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def pixPerMat(image):
    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 0, 60)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    # loop over the contours individually
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue
        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        
        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        
        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        if pixelsPerMetric is None:
            pixelsPerMetric = dB/ 7

    return pixelsPerMetric
    
#This method work with colorimage and return pixelPerMatrix
def countError(image, imagecolor):    
    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    sobel_x = cv2.Sobel(gray, -1, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, -1, 0, 1, ksize=5)
    edged = cv2.addWeighted(sobel_x, 1, sobel_y, 1, 0)
    #edged = cv2.Canny(gray, 0, 60)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    #test = thresh_sauvola(gray,25)
    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = pixPerMat(imagecolor)
    allerror1 = 0
    allerror2 = 0
    allerror3 = 0
    allerror4 = 0
    # loop over the contours individually
    
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 200:
            continue
        if cv2.contourArea(c) <= 700:
            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            # loop over the original points and draw them
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
        
            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            # draw the midpoints on the image
            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
            # draw lines between the midpoints
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)
        
            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # compute the size of the object
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric
            if( dimA > dimB): #เส้นแนวตั้ง
                point = dimA/0.05
                point = round(point)
                if (point > 0):
                    point = 1
                    cv2.putText(imagecolor, "{:.1f}".format(point),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 0, 0), 2)
                    allerror1 = allerror1 + point
            if( dimA < dimB):  #เส้นแนวนอน
                point = dimB/0.05
                point = round(point)
                if (point > 0):
                    point = 1
                    cv2.putText(imagecolor, "{:.1f}".format(point),(int(trbrX + 20), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 0, 0), 2)
                    allerror2 = allerror2 + point
        if cv2.contourArea(c) > 700:
            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            # loop over the original points and draw them
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
        
            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            # draw the midpoints on the image
            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
            # draw lines between the midpoints
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)
        
            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # compute the size of the object
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric
            if( dimA > dimB): #เส้นแนวตั้ง
                point = dimA/0.2
                point = round(point)
                if (point > 0):
                    cv2.putText(imagecolor, "{:.1f}".format(point),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 0, 0), 2)
                    allerror3 = allerror3 + point
            if( dimA < dimB):  #เส้นแนวนอน
                point = dimB/0.2
                point = round(point)
                if (point > 0):
                    cv2.putText(imagecolor, "{:.1f}".format(point),(int(trbrX + 20), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 0, 0), 2)
                    allerror4 = allerror4 + point
    allerror = allerror1 + allerror2 + allerror3 + allerror4

    return imagecolor , allerror

def resize_img(image):
    img = image
    # resize image
    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resize_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    return resize_image

def checkError(img):
    dila = 3
    ero = 3
    upper_green = np.array([50, 255, 50], dtype = "uint8")
    lower_green = np.array([0, 255, 0], dtype = "uint8")
    mask_green = cv2.inRange(img.copy(), lower_green, upper_green)
    #mask_green = cv2.dilate(mask_green,(5,5),iterations = dila)
    cnts_g = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts_g = cnts_g[0] if len(cnts_g) == 2 else cnts_g[1]

    upper_red = np.array([50, 50, 255], dtype = "uint8")
    lower_red = np.array([0, 0, 255], dtype = "uint8")
    mask_red = cv2.inRange(img.copy(),lower_red , upper_red)
    mask_red = cv2.dilate(mask_red,(5,5),iterations = dila)
    mask_red = cv2.erode(mask_red,(5,5),iterations = ero)
    cnts_r = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts_r = cnts_r[0] if len(cnts_r) == 2 else cnts_r[1]
    #cv2.imwrite('D:/project-final/pattern1-testlog/%s/%s-mask_green.png' % (nfolder, nn) , mask_green)
    return mask_green

def process(file):

    image_gray = cv2.cvtColor(file,cv2.COLOR_BGR2GRAY)

    #ANSWER SHEET 
    image_thresh = thresh_sauvola(image_gray, 73)
    #cv2.imwrite('D:/project-final/pattern1-testlog/%s/%s-2-image_thresh.png' % (nfolder, nn) , image_thresh)

    cnt = find_border(image_thresh)
    ans_noboder = delete_border(image_thresh, cnt)

    #---- make pattern's line (เส้นประ+เส้นดินสอ) thickness ----#
    kernel = np.ones((7,7),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(ans_noboder,kernel,iterations = 3)
    dilation = cv2.erode(dilation,kernel,iterations = 1)
    
    #PATTERN SHEET  --> เส้นไม่สัมผัสเส้นประ และ เส้นเกิน
    pattern1 = make_pattern(file , cnt, 10, 10) 
    pattern1_re = recheck_pattern(file,cnt, 10, 10)
    surp1 = cv2.subtract(pattern1 , pattern1_re)
    pattern1_final = cv2.subtract(pattern1,surp1)

    #PATTERN SHEET  --> ช่องว่างระหว่างเส้นประ
    pattern2 = make_pattern(file , cnt, 14, 12)  
    pattern2_re = recheck_pattern(file,cnt, 10, 15)      # --> (image , point, dilation, erosion) 
    surp2 = cv2.subtract(pattern2 , pattern2_re)
    pattern2_final = cv2.subtract(pattern2,surp2)

    #Find error
    error = cv2.subtract(ans_noboder,pattern1_final) #--> เส้นไม่สัมผัสเส้นประ และ เส้นเกิน
    error2 = cv2.subtract(pattern2_final,dilation)  #new--> ช่องว่างระหว่างเส้นประ
    #error = cv2.subtract(pattern2_final,dilation)
    #bitwiseOr = cv2.bitwise_or(error, error2)
    error = cv2.cvtColor(error ,cv2.COLOR_GRAY2BGR)
    error2 = cv2.cvtColor(error2 ,cv2.COLOR_GRAY2BGR)

    #---- coloring error line with difference color ----#   
    #drawerror = draw_error(image_color,error1)
    #drawerror2 = draw_error2(image_color,error2)

    list = draw_error(file,error)
    list = draw_error2(file,error2)
    #segment_im(pattern1_final, image_color) #-->image with 2 error type without countingnumber
    #maskG = checkError(list)
    #maskG = cv2.cvtColor(maskG ,cv2.COLOR_GRAY2BGR)
    allerr = cv2.bitwise_or(error, error2)

    #showimageAllerrorpoint ,allerrorpoint =  countError(error,image_color)
    showimageAllerrorpoint ,allerrorpoint =  countError(allerr, file)
    #cv2.imwrite('D:/project-final/pattern1-testlog/%s/%s-3-showimageAllerrorpoint.png' % (nfolder, nn) , showimageAllerrorpoint)
    print(allerrorpoint)
    return showimageAllerrorpoint , allerrorpoint
    
    #return show
