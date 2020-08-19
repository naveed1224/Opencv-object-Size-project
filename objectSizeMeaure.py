import cv2
import numpy as np
import time

#webcam toggle
webcam = False
path = '4.jpg'

def getContours(img, cThr=[100,100], showCanny = False, minArea = 1000, filter = 0, draw=False):
    #apply gray scale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #apply grayscale, with kernal of (5,5)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    #apply canny edge detection, threshold can be overridden but default 100,100
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])

    #image Dialet kernal
    kernel = np.ones((5,5))

    imgDial = cv2.dilate(imgCanny, kernel, iterations=4)

    imgThre = cv2.erode(imgDial, kernel, iterations=2)

    #show picture
    if showCanny:
        cv2.imshow('Canny', imgThre)

    #All contours in the image
    contours, hierarchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #loop through all contours and get contour area
    finalContour = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            bbox = cv2.boundingRect(approx)

            #filtering out rectangles

            if filter > 0:
                if len(approx) == filter:
                    finalContour.append((len(approx), area, approx, bbox, i))
            else:
                finalContour.append((len(approx), area, approx, bbox, i))
    finalContour = sorted(finalContour, key = lambda x:x[1], reverse=True)

    if draw:
        for con in finalContour:
            cv2.drawContours(img, con[4], -1, (0,0, 255), 3)
    return img, finalContour

def reOrder(pts):
    #print(pts.shape)
    ptsNew = np.zeros_like(pts)
    pts = pts.reshape((4,2))
    add = pts.sum(1)
    #top left corner of page
    ptsNew[0] = pts[np.argmin(add)]
    #bottom right corner of page
    ptsNew[3] = pts[np.argmax(add)]
    diff = np.diff(pts, axis=1)
    #top right corner of page
    ptsNew[1] = pts[np.argmin(diff)]
    #bottom left corner of page
    ptsNew[2] = pts[np.argmax(diff)]
    return ptsNew



def warpImage(img, points, w, h, pad=20):
    #print(points)
    points = reOrder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad, pad:imgWarp.shape[1]-pad]
    return imgWarp

def measureDist(pts1, pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5

if webcam:
    cap = cv2.VideoCapture(0)
    cap.set(10, 160)
    cap.set(3, 1920)
    cap.set(4,1080)
    scale = 3
    wP = 210
    hP = 270


while True:
    if webcam:
        success, img = cap.read()
    else:
        img =  cv2.imread(path)

    img, finalContour = getContours(img, showCanny = True, draw=True, minArea=50000, filter=4)

    if len(finalContour) != 0:
        biggest = finalContour[0][2]
        #print(biggest)
        imgWarp = warpImage(img, biggest, 210 *3, 297 * 3)
        cv2.imshow('A4 paper', imgWarp)
        #finding contours within existing main contour
        img2, finalContour2 = getContours(imgWarp, showCanny = True, draw=False, minArea=2000, filter=4, cThr=[50,50])


        if len(finalContour) != 0:
            for obj in finalContour2:
                cv2.polylines(img2, [obj[2]], True, (0, 255, 0), 2)
                nPoints = reOrder(obj[2])
                nW = round(measureDist(nPoints[0][0]//3, nPoints[1][0]//3)/10,1)
                nH = round(measureDist(nPoints[0][0]//3, nPoints[2][0]//3)/10,1)

                #drawing lines on page

                cv2.arrowedLine(img2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(img2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(img2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(img2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)

        cv2.imshow('cards', img2)




    img = cv2.resize(img, (0,0), None, 0.5, 0.5)
    cv2.imshow('Original', img)
    cv2.waitKey(1)
