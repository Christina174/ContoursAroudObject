import cv2
import numpy as np
import copy

drawing = False
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

    if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True

            currentContour = np.append(currentContour, [x,y]) # add point coords to current array
            currentContour = currentContour.reshape ((-1,2))

            if currentContour.shape[0] > 1: # if polyline`s points in array >1 draw polyline
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
    
    #stop draw current contours and add points to array contours
    elif event == cv2.EVENT_RBUTTONDOWN: 
            drawing = False 
            if currentContour.shape[0] > 2:
                arrayContour.append(currentContour)
            pic = drawContours(img, arrayContour, finishColor)
            currentContour = np.array ([])

# Create a black image, a window and bind the function to window

cv2.namedWindow('image', flags= cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE ) # settings params window with image size and without dropdown menu
cv2.setMouseCallback('image',mousePosition)

while(1):
    cv2.imshow('image',pic) #
    k = cv2.waitKey(30)
    if k == 27: # Esc
        break

cv2.destroyAllWindows()
