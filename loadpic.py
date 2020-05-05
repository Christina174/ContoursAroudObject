import pandas as pd
import cv2

pics = pd.read_csv('list.csv', header=None)
i=0
if len(pics)==0:
    print('List have length = 0')
    pass

while(1):
    img = cv2.imread(pics.iloc[i][0])
    r = pics.iloc[i][0]
    cv2.imshow(r, img)
    k = cv2.waitKey(0)
    if k == 97: # Left
        i-=1
    if k == 100: # Right
        i+=1
    if k == 27: # Esc
        break
    if i >= len(pics) or i <= -len(pics): # new cycle 
        i = 0
    cv2.destroyWindow(r)

cv2.destroyAllWindows()
