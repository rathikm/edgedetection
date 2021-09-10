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

def custom_roll(arr):
    dim = arr.shape
    rown = dim[0]
    coln = dim[1]
    collen = rown - 2
    col1 = arr[1:rown-1,0]
    col2 = arr[1:rown-1,coln-1]
    row1 = arr[0,:]
    row2 = arr[rown-1,:]
    row2 = np.flip(row2)
    col1 = col1.reshape(1, -1)[0]
    col2 = col2.reshape(1, -1)[0]
    col1 = np.flip(col1)
    flat = np.concatenate((row1, col2, row2, col1))
    flat = np.roll(flat, 1)
    arr[0,:] = flat[0:coln]
    arr[1:rown-1,coln-1] = flat[coln:coln+collen]
    arr[rown-1,:] = np.flip(flat[coln+collen:2*coln+collen])
    arr[1:rown-1,0] = np.flip(flat[2*coln+collen:2*coln+2*collen])


def create_robinson():
    masks = []
    
def get_outer_length(arr):
    dim = arr.shape
    rown = dim[0]
    coln = dim[1]
    return 2 * rown + 2 * (coln-2)


path = "image.jpg" #replace with image name
img = cv.imread(path)
mimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gimg = resize(mimg, 15)
dim = gimg.shape
dimr = dim[0]
dimc = dim[1]
sf = create_sobel()
length = get_outer_length(sf)
maps = np.zeros((length, dimr, dimc))
for i in range(0,length):
    custom_roll(sf)
    edmap = detection_pass(gimg, sf)
    emap = convert_img(edmap, -1, 1)
    print(i)
    maps[i] = emap
edgemap = np.zeros_like(gimg)
for i in maps:
    print(i)
    edgemap = edgemap + i ** 2
edgemap = edgemap ** (1/maps.shape[0])
while True:
    cv.imshow("Edgemap", edgemap)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()


