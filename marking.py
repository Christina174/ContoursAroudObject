import cv2
import numpy as np
import copy
import pandas as pd



#arrayContour = [] # array finished contours

indexPics = 0 # index current image




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
This function find minimal distance between point and segment in 2-D

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
    array: list of contours
    point: coordinates of point

Returns:
    List of minimal distances from point to each contours.
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




import json

"""
This function read json-file with structure:
# json file example
[
    {
            "contour" : 
                [
                    {
                        "x" : 0,
                        "y" : 0
                    },
                    {
                        "x" : 10,
                        "y" : 10
                    }
                ]
    },
    {
    }
]

and get list of contours.

Args:
    filename: name of json-file 

Returns:
    List of contours. Element of list is contour. Every contour have a type numpy.ndarray, dimention (n, 2), n-number points in contour.  
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
This function save contour in text file as json

Args:
    filename: name of json-file
    arrayContour: array of contours
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
This function load image by index and create new window, loading early saved contours for this image

Args:
    indexPics: index of new image

Returns:
    filename - name of json-file
    numImage - name image
    pic - deepcopy of source image with drawn contours
    arrayContour - array of contours
    img - loaded image
"""
#def createImage(indexPics): # load data and create new window
    #global drawing
    #global crossline
    #global selectedContour
    #global indexSelectedContour
    #global currentContour
    
    #drawing = False
    #crossline = False
    #selectedContour = False # flag mean that we choose contour for delete
    #indexSelectedContour = -1 # index of contour that we choose
    #currentContour = np.array ([]) # array  points of polygon that haven't finished drawing
    #numImage = pics.iloc[indexPics][0]
    #img = cv2.imread(numImage)
    #filename = numImage+'.json' 
    #arrayContour = loadJson(filename)
    #pic = drawContours(img, arrayContour, finishColor)
    #cv2.namedWindow(numImage, flags= cv2.WINDOW_GUI_NORMAL ) #| cv2.WINDOW_AUTOSIZE settings params window without dropdown menu and with image size
    #cv2.setMouseCallback(numImage,mousePosition)
    #return filename, numImage, pic, arrayContour, img
    

tempColor = (0,255,0) # color of polygon that haven't finished drawing
finishColor = (0,255,255) # color finished contours
thickness = 3 # thickness of line contours
thDistance = 5 # min distance from click point to contour

   
"""
This function load image by index and create new window, loading early saved contours for this image

Args:
    indexPics: index of new image

Returns:
    filename - name of json-file
    numImage - name image
    pic - deepcopy of source image with drawn contours
    arrayContour - array of contours
    img - loaded image
"""
class Contour:
#"""saveJson
#This function draw poligons on image

#Args:
    #image: source image
    #contours: array finished contours
    #color: color finished contours

#Returns:
    #Deep copy of source image with finished contours
#"""

    def drawContours(self): 
        self.pic = copy.deepcopy(self.img)
        cv2.polylines (self.pic, self.arrayContour, True, finishColor, thickness)     
    
    def __init__(self, imageFileName):
        self.imageFileName = imageFileName
        self.drawing = False
        self.crossline = False
        self.selectedContour = False # flag mean that we choose contour for delete
        self.indexSelectedContour = -1 # index of contour that we choose
        self.currentContour = np.array ([]) # array  points of polygon that haven't finished drawing
        self.img = cv2.imread(imageFileName)
        self.jsonFileName = imageFileName+'.json' 
        self.arrayContour = loadJson(self.jsonFileName)
        self.pic = None
        
        self.drawContours()
        cv2.namedWindow(imageFileName, flags= cv2.WINDOW_GUI_NORMAL ) #| cv2.WINDOW_AUTOSIZE settings params window without dropdown menu and with image size
        cv2.setMouseCallback(imageFileName, self.mousePosition)

    def destroyImage(self):
        # save data and destroy window
        cv2.destroyWindow(self.imageFileName)
        saveJson(self.jsonFileName, self.arrayContour)
        
    def deleteContour(self):
        if self.selectedContour:
            self.arrayContour.pop(self.indexSelectedContour)
            self.selectedContour = False
            drawContours()
            

        
    # mouse callback function
    def mousePosition(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
                drawContours()
                if self.currentContour.shape[0] == 0 and len(self.arrayContour)>0: # if the first click and array of finished contours have coordinates, we test distance between point and contours
                    listDist = distanceToContours(self.arrayContour, np.array([x,y]))
                    md = min(listDist)
                    self.indexSelectedContour = listDist.index(md)
                    if md <= thDistance: # if minimal distance from point to contour smaller threshhold => choose contour
                        self.selectedContour = True
                        cv2.polylines (pic, [self.arrayContour[self.indexSelectedContour]], True , (0,0,255), thickness)
                        cv2.putText(pic, "press 'Del' to delete", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                        return
                    
                self.drawing = True
                if self.crossline == True:
                    return
                
                self.currentContour = np.append(self.currentContour, [x,y]) # add point coords to currentContour
                self.currentContour = self.currentContour.reshape ((-1,2))

                if self.currentContour.shape[0] > 1:  # if polyline`s points in array >1, draw polyline
                    self.currentContour = self.currentContour.astype(np.int32)
                    cv2.polylines (self.pic, [self.currentContour], False , tempColor, thickness) #
                    
        if event == cv2.EVENT_MOUSEMOVE:
                if self.currentContour.shape[0] > 0 and self.drawing == True:
                        currentContour2 = np.append(self.currentContour, [x,y])
                        currentContour2 = currentContour2.reshape ((-1,2))
                        currentContour2 = currentContour2.astype(np.int32)
                        drawContours() #pic = 
                        cv2.polylines (self.pic, [currentContour2], False , tempColor, thickness) #
                #test on cross lines
                if self.currentContour.shape[0] > 2:
                    X1,Y1,X2,Y2 = (self.currentContour[-1][0], self.currentContour[-1][1], x, y) #current line
                    for i in range(len(self.currentContour)-2):
                        X3,Y3,X4,Y4 = (self.currentContour[i][0], self.currentContour[i][1], self.currentContour[i+1][0], self.currentContour[i+1][1]) #previous line
                        cL1 = ((X3-X1)*(Y2-Y1)-(Y3-Y1)*(X2-X1))*((X4-X1)*(Y2-Y1)-(Y4-Y1)*(X2-X1))
                        cL2 = ((X1-X3)*(Y4-Y3)-(Y1-Y3)*(X4-X3))*((X2-X3)*(Y4-Y3)-(Y2-Y3)*(X4-X3))
                        if cL1<=0 and cL2<=0:
                            cv2.putText(self.pic, "Crossing lines", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                            self.crossline = True
                            break
                        else:
                            self.crossline = False
                    
        #stop draw current contours and add points to array contours
        elif event == cv2.EVENT_RBUTTONDOWN: 
                self.drawing = False
                self.crossline = False
                self.selectedContour = False
                if self.currentContour.shape[0] > 2:
                    self.arrayContour.append(self.currentContour)
                drawContours() #pic = 
                self.currentContour = np.array ([])

    def imShow(self):
        print(self.pic.shape)
        cv2.imshow(self.imageFileName, self.pic)

    
pics = pd.read_csv('list.csv', header=None) # load list images
if len(pics)==0:
    print('List have length = 0')
    pass

newIndexPic = indexPics
#filename, numImage, pic, arrayContour, img = createImage(indexPics) 

contour = Contour(pics.iloc[indexPics][0]) #create the first image

while(1):
    contour.imShow()
    k = cv2.waitKey(30)
    if k == 27: # Esc
        break
    if k == 255: # Del
        contour.deleteContour()
    if k == 97: # Left bottom 'a'
        newIndexPic = indexPics -1
    if k == 100: # Right bottom 'd'
        newIndexPic = indexPics + 1
    if newIndexPic >= len(pics) or newIndexPic <= -len(pics): # new cycle of list with pics
        newIndexPic = 0
    if newIndexPic != indexPics:
        contour.destroyImage()
        indexPics = newIndexPic
        contour = Contour(pics.iloc[indexPics][0]) #create next image
        

cv2.destroyAllWindows()
