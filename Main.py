import cv2
from preprocessing import Prep as p
import numpy as n

obj=p.Prep('melanoma.jpg')
#obj.OtsuAutoThresh(np.array([[23,13],[127,54],[56,98]]))
#arr = n.array([(23,),(9,),(13,),(27,),(53,),(47,),(92,),(2,)],dtype=n.dtype([('val',n.uint8)]))
#print(arr)
#p.quickSort(arr,0,arr.size-1)
#print(arr)
obj.imBinarize(obj.getInvrtGrayImg())
cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.imshow('img',obj.getBinaryImg())
cv2.waitKey(0)
#img = cv2.imread('melanoma.jpg',cv2.IMREAD_GRAYSCALE)
#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#cv2.imshow('image',obj.getColorPlates(obj.getActImg(),'B'))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imwrite('melgray.png',img)
#print(img)
#print(img.shape)
#print(img[674,1023])

#(r,c) = img.shape

#for x in range(r):
    #for y in range(c):
        #img[x,y]=np.invert(img[x,y])
#for x in img:
    #x = 255 - x
    #x = np.invert(x)
    #print(type(x))
    #print(x)
    #print("asd")
    #for y in x:
        #print(y)
        #y[0] = 0
        #y[1] = 0


#print(img)

#cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
#cv2.imshow('image1',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()