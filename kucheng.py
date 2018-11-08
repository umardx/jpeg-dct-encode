import cv2
import numpy as np
import pandas as pd


def write_xlsx(arr, path):
    return pd.DataFrame(arr).to_excel(path)

def load_img(img):
    return cv2.imread(img)

def rgb2yuv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

def split_channel(img):
    return cv2.split(img)
    
def blockshaped(arr, nrows, ncols):
    """
    # https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    
    c = np.arange(24).reshape((4,6))
    print(c)
    # [[ 0  1  2  3  4  5]
    #  [ 6  7  8  9 10 11]
    #  [12 13 14 15 16 17]
    #  [18 19 20 21 22 23]]
    """
    
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))