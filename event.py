import cv2
import numpy as np
import copy


img = np.zeros((512,512,3), np.uint8)
imgclone = copy.deepcopy(img)
pts = np.array ([])
#pts = []
#np.int32
ix,iy = -1,-1
# mouse callback function
def mousePosition(event,x,y,flags,param):
    global imgclone
    imgclone = copy.deepcopy(img)
    
    global pts
    
    if event == cv2.EVENT_LBUTTONDOWN:
        pts = np.append(pts, [x,y])
        #print(pts.shape)
        ix,iy = x,y
        #print ([x,y])
        #pts = pts.reshape((-1,1,2))

        pts = pts.reshape ((-1,2))
        #print(pts)
        #print(pts.shape[0])
        if pts.shape[0] > 1:
            pts = pts.astype(np.int32)
            cv2.polylines (imgclone, [pts], False , (0,255,255))
    if event == cv2.EVENT_MOUSEMOVE:
        #pts2 = copy.deepcopy(pts)
        if pts.shape[0] > 0:
            pts2 = np.append(pts, [x,y])
            pts2 = pts2.reshape ((-1,2))
            pts2 = pts2.astype(np.int32)
            cv2.polylines (imgclone, [pts2], False , (0,255,0))
    if event == cv2.EVENT_RBUTTONDOWN:
    #if len(pts)==2:
        #np.append(pts, pts[0])
        #np.append(pts, np.int32)
        #print (x,y)
        #print(pts)
        #cv2.rectangle(img,pts[0],pts[1], (0,255,0),1)
        #pts = pts.reshape ((- 1,1,2))
        cv2.polylines (img, [pts], True , (0,255,255), 1)



# Create a black image, a window and bind the function to window

cv2.namedWindow('image')
cv2.setMouseCallback('image',mousePosition)

while(1):
    cv2.imshow('image',imgclone)
    k = cv2.waitKey(30)
    if k == 27:
        break

cv2.destroyAllWindows()
