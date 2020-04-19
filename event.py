import cv2
import numpy as np
import copy

drawing = False
crossline = False
#img = np.zeros((512,512,3), np.uint8)
img = cv2.imread('im2.jpg')
pic = copy.deepcopy(img) # copy source image for drawing contours
currentContour = np.array ([]) # array  points of polygon that haven't finished drawing
arrayContour = [] # array finished contours
tempColor = (0,255,0)
finishColor = (0,255,255)
thickness = 3 # thickness of line contours


# draw ready poligons
def drawContours(image, contours, color): 
    imgCopy = copy.deepcopy(image)
    cv2.polylines (imgCopy, contours, True, color, thickness) 
    return imgCopy
    
# mouse callback function
def mousePosition(event,x,y,flags,param):
    global currentContour
    global drawing
    global pic
    global arrayContour
    global crossline
    
    if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            if crossline == True:
                return
            
            currentContour = np.append(currentContour, [x,y]) # add point coords to current array
            currentContour = currentContour.reshape ((-1,2))



            if currentContour.shape[0] > 1:  # if polyline`s points in array >1 draw polyline
                currentContour = currentContour.astype(np.int32)
                pic = drawContours(img, arrayContour, finishColor)
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
            if currentContour.shape[0] > 2:
                arrayContour.append(currentContour)
            pic = drawContours(img, arrayContour, finishColor)
            currentContour = np.array ([])
            #print(arrayContour)

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

filename = 'arrayContours.json'
try:
    with open(filename) as f:
        arrayContourJson = json.load(f)
        #print(len(arrayContourJson))
        #arrayContour = []
        for element in arrayContourJson:
            #print(element)
            pts = np.array ([])
            for point in element["contour"]:
                pts = np.append(pts, [point["x"],point["y"]])
                
            pts = pts.astype(np.int32)
            pts = pts.reshape ((-1,2))
            arrayContour.append(pts)
            #print(arrayContour)
        pic = drawContours(img, arrayContour, finishColor)
except IOError:
    pass

cv2.namedWindow('image', flags= cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE ) # settings params window with image size and without dropdown menu
cv2.setMouseCallback('image',mousePosition)

while(1):
    cv2.imshow('image',pic) #
    k = cv2.waitKey(30)
    if k == 27: # Esc
        break

cv2.destroyAllWindows()


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
    
    
with open('arrayContours.json', 'w') as f:
    f.write(json.dumps(arrayContourJson, indent=4))

