import pandas as pd
import cv2

#imga = cv2.imread('im2.jpg')

pics = pd.read_csv('list.csv', header=None)
i=0
if len(pics)==0:
    print('List have length = 0')
    pass
cv2.namedWindow('image', flags= cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE ) # settings params window with image 

while(1):
    img = cv2.imread(pics.iloc[i][0])
    cv2.imshow('image', img)
    k = cv2.waitKey(30)
    print(k)
    if k == 81: # Left
        #if i==0:
        i-=1
        #i=pics[-i]
    if k == 83: # Right
        i+=1
        #i=pics[-i]
    if k == 27: # Esc
        break


cv2.destroyAllWindows()
