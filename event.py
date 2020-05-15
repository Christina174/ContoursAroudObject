import cv2
import numpy as np
import copy
import pandas as pd

drawing = False
crossline = False

currentContour = np.array ([]) # array  points of polygon that haven't finished drawing
#arrayContour = [] # array finished contours
tempColor = (0,255,0) # color of polygon that haven't finished drawing
finishColor = (0,255,255) # color finished contours
thickness = 3 # thickness of line contours
thDistance = 5 # min distance from click point to contour
indexSelectedContour = -1 # index of contour that we choose
selectedContour = False # flag mean that we choose contour for delete
indexPics = 0 # index current image

pics = pd.read_csv('list.csv', header=None) # load list with pics path
if len(pics)==0:
    print('List have length = 0')
    pass


"""
This function draw poligons on image

Args:
    image: source image
    contours: array finished contours
    color: color finished contours

Returns:
    Deep copy of source image with finished contours
"""
def drawContours(image, contours, color): 
    imgCopy = copy.deepcopy(image)
    cv2.polylines (imgCopy, contours, True, color, thickness) 
    return imgCopy

"""
This function calculates the square of the vector length

Args:
    v: vector. Type numpy.ndarray.

Returns:
    Square of the vector length. Type numpy.float64
"""
def magnitude(v):
    return np.sum(np.fromiter((vi ** 2 for vi in v), float))

"""
This function find minimal distance between point and segment

Args:
    a: Coordinates of point A of the segment AB. Type numpy.ndarray.
    b: Coordinates of point B of the segment AB. Type numpy.ndarray.
    p: Coordinates of point, distance for that find.

Returns:
    Minimal distance between point and segment. Type numpy.float64
"""
def minDistance(a, b, p): # a, b - coords of line, p - coords of click point
    ab = b-a
    bp = b-p
    pa = p-a
    dAB = magnitude(ab)
    dBP = magnitude(bp)
    dPA = magnitude(pa)
    if dPA > (dBP + dAB) or dBP > (dPA + dAB):
        if dPA < dBP:
            minDistance = np.sqrt(dPA)
        else:
            minDistance = np.sqrt(dBP)
    else:
        minDistance = abs(ab[0]*p[1] - ab[1]*p[0] + b[1]*a[0] - b[0]*a[1])/np.sqrt(dAB)
    return minDistance

"""
This function generates list of minimal distances from point to each contours

Args:
    array: list of finished contours
    point: coordinates of current point

Returns:
    List of minimal distances from point to each contours
"""
def distanceToContours(array, point):
    distToContours = [] # list of minimal distances from point to each contour
    for contour in array:
        distToSegments = [] # list of distances to each line of contour
        for i in range(contour.shape[0]-1): # determining the distance from the point to all segments of the contour
            m = minDistance(contour[i], contour[i+1], point)
            distToSegments.append(m)
        distToSegments.append(minDistance(contour[-1], contour[0], point)) # minimal distance from a point to the segment that closes the contour
        distToContours.append(min(distToSegments)) # choose minimal distance to contour
    return distToContours


# mouse callback function
def mousePosition(event,x,y,flags,param):
    global currentContour
    global drawing
    global pic
    global arrayContour
    global crossline
    global indexSelectedContour
    global selectedContour

    
    if event == cv2.EVENT_LBUTTONDOWN:
            pic = drawContours(img, arrayContour, finishColor)
            if currentContour.shape[0] == 0 and len(arrayContour)>0: # if the first click and array of finished contours have coordinates, we test distance between point and contours
                listDist = distanceToContours(arrayContour, np.array([x,y]))
                md = min(listDist)
                indexSelectedContour = listDist.index(md)
                if md <= thDistance: # if minimal distance from point to contour smaller threshhold => choose contour
                    selectedContour = True
                    cv2.polylines (pic, [arrayContour[indexSelectedContour]], True , (0,0,255), thickness)
                    cv2.putText(pic, "press 'Del' to delete", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                    return
                
            drawing = True
            if crossline == True:
                return
            
            currentContour = np.append(currentContour, [x,y]) # add point coords to currentContour
            currentContour = currentContour.reshape ((-1,2))

            if currentContour.shape[0] > 1:  # if polyline`s points in array >1, draw polyline
                currentContour = currentContour.astype(np.int32)
                cv2.polylines (pic, [currentContour], False , tempColor, thickness) #
                
    if event == cv2.EVENT_MOUSEMOVE:
            if currentContour.shape[0] > 0 and drawing == True:
                    currentContour2 = np.append(currentContour, [x,y])
                    currentContour2 = currentContour2.reshape ((-1,2))
                    currentContour2 = currentContour2.astype(np.int32)
                    pic = drawContours(img, arrayContour, finishColor)
                    cv2.polylines (pic, [currentContour2], False , tempColor, thickness) #
            #test on cross lines
            if currentContour.shape[0] > 2:
                X1,Y1,X2,Y2 = (currentContour[-1][0], currentContour[-1][1], x, y) #current line
                for i in range(len(currentContour)-2):
                    X3,Y3,X4,Y4 = (currentContour[i][0], currentContour[i][1], currentContour[i+1][0], currentContour[i+1][1]) #previous line
                    cL1 = ((X3-X1)*(Y2-Y1)-(Y3-Y1)*(X2-X1))*((X4-X1)*(Y2-Y1)-(Y4-Y1)*(X2-X1))
                    cL2 = ((X1-X3)*(Y4-Y3)-(Y1-Y3)*(X4-X3))*((X2-X3)*(Y4-Y3)-(Y2-Y3)*(X4-X3))
                    if cL1<=0 and cL2<=0:
                        cv2.putText(pic, "Crossing lines", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                        crossline = True
                        break
                    else:
                        crossline = False
                
    #stop draw current contours and add points to array contours
    elif event == cv2.EVENT_RBUTTONDOWN: 
            drawing = False
            crossline = False
            selectedContour = False
            if currentContour.shape[0] > 2:
                arrayContour.append(currentContour)
            pic = drawContours(img, arrayContour, finishColor)
            currentContour = np.array ([])


# json file example
#[
    #{
            #"class" : "cruassan",
            #"contour" : 
                #[
                    #{
                        #"x" : 0,
                        #"y" : 0
                    #},
                    #{
                        #"x" : 10,
                        #"y" : 10
                    #}
                #]
    #},
    #{
    #}
#]

import json

"""
This function open json-file named as current image with coordinates "x": , "y":  contours and add to array arrayContour

Args:
    filename: json-file named as image

Returns:
    List of contours
"""
def loadJson(filename):
    arrayContour = []
    try:
        with open(filename) as f:
            arrayContourJson = json.load(f)
            for element in arrayContourJson:
                pts = np.array ([])
                for point in element["contour"]:
                    pts = np.append(pts, [point["x"],point["y"]])
                    
                pts = pts.astype(np.int32)
                pts = pts.reshape ((-1,2))
                arrayContour.append(pts)
            
    except IOError:
        pass
    return arrayContour

"""
This function create arrayContourJson - list of coordinates "x": , "y":  contours and write it in file

Args:
    filename: json-file named as image
    arrayContour: array of finished contours

Returns:
    In final, open json-file named as image and write
"""
def saveJson(filename, arrayContour):
    arrayContourJson = []
    for contour in arrayContour:
        countourJson = []
        for i in range(contour.shape[0]):
            point = dict()
            point["x"] = int(contour[i,0])
            point["y"] = int(contour[i,1])
            countourJson.append(point)
        
        element = dict()
        element["contour"] = countourJson
        arrayContourJson.append(element)
        
    with open(filename, 'w') as f:
        f.write(json.dumps(arrayContourJson, indent=4))

"""
This function choose new image from csv-file and create window, loading early saved contours

Args:
    indexPics: index of new image

Returns:
    Filename for saving json-file, numImage - name current image, pic-copy of image with drawn contours, arrayContour-array of finished contours, img - source image
"""
def createImage(indexPics): # load data and create new window
        numImage = pics.iloc[indexPics][0]
        img = cv2.imread(numImage)
        filename = numImage+'.json' 
        arrayContour = loadJson(filename)
        pic = drawContours(img, arrayContour, finishColor)
        cv2.namedWindow(numImage, flags= cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE ) # settings params window with image size and without dropdown menu
        cv2.setMouseCallback(numImage,mousePosition)
        return filename, numImage, pic, arrayContour, img
    

newIndexPic = indexPics
filename, numImage, pic, arrayContour, img = createImage(indexPics) #create the first image

while(1):
    cv2.imshow(numImage,pic) #
    k = cv2.waitKey(30)
    if k == 27: # Esc
        break
    if k == 255: # Del
        if selectedContour:
            arrayContour.pop(indexSelectedContour)
            selectedContour = False
            pic = drawContours(img, arrayContour, finishColor)
    if k == 97: # Left bottom 'a'
        newIndexPic = indexPics -1
    if k == 100: # Right bottom 'd'
        newIndexPic = indexPics + 1
    if newIndexPic >= len(pics) or newIndexPic <= -len(pics): # new cycle of list with pics
        newIndexPic = 0
    if newIndexPic != indexPics:
        # save data and destroy window
        cv2.destroyWindow(numImage)
        saveJson(filename, arrayContour)
        
        indexPics = newIndexPic
        filename, numImage, pic, arrayContour, img = createImage(indexPics) #create next image
        

cv2.destroyAllWindows()
