import cv2 as cv
import numpy as np
import sys


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


def convert_img(img, rmin, rmax):
    maxval = np.amax(img)
    minval = np.amin(img)
    print(maxval)
    print(minval)
    converter = lambda x: translate(x, minval, maxval, rmin, rmax)
    newimg = np.copy(img)
    convvec = np.vectorize(converter)
    nimg = convvec(newimg)
    return nimg

def detection_pass(img, filter):
    dim = img.shape
    rown = dim[0]
    coln = dim[1]
    fdim = filter.shape
    edgemap = np.zeros((rown, coln))
    for r in range(0, rown-3):
        for c in range(0, coln-3):
            window = img[r:r+3, c:c+3]
            ewindow = window * filter
            sum = summer_2D(ewindow)
            edgemap[r+1,c+1] = sum
    #print(edgemap)
    return edgemap        

def summer_2D(arr):
    row = np.sum(arr, axis=1)
    final = np.sum(row)
    return final

def create_sobel():
    sf = np.zeros((3,3))
    sf[0,0] = -1    
    sf[0,2] = 1
    sf[1,0] = -2
    sf[1,2] = 2
    sf[2,0] = -1
    sf[2,2] = 1
    return sf

def resize(img, percent):
    p = percent
    w = img.shape[1] * percent/100
    h = img.shape[0] * percent/100
    dim = (int(w), int(h))
    scaledimg = cv.resize(src=img, dsize=dim, interpolation=cv.INTER_AREA)
    return scaledimg

np.set_printoptions(threshold=sys.maxsize)
path = "image.jpg" #replace with name of image
img = cv.imread(path)
mimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gimg = resize(mimg, 90)
#print(np.amax(gimg))
#ggimg = np.zeros_like(gimg)
#cv.normalize(gimg, ggimg, 0, 1, cv.NORM_MINMAX)

#cv.normalize(gimg, ggimg, 0, 255, cv.NORM_MINMAX)
#print(ggimg)
#SOBEL_FILTER = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
SOBEL_FILTER_X = create_sobel()
SOBEL_FILTER_Y = np.transpose(create_sobel())
edgemapX = detection_pass(gimg, SOBEL_FILTER_X)
emapX = convert_img(edgemapX, -1, 1)
edgemapY = detection_pass(gimg, SOBEL_FILTER_Y)
emapY = convert_img(edgemapY, -1, 1)
#emapX[0:60, 0:60] = 0.8
#edgemapY = detection_pass(gimg, SOBEL_FILTER_Y)
edgemap = (emapX ** 2 + emapY ** 2)**0.5
#print(edgemap)
#print(edgemapX)
#edgemap = resize(emap, 30)
#a = np.array([4,5,3,7]).reshape(2,2)
#print(summer_2D(a))
while True:
    cv.imshow("Edgemap", edgemap)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()
    
