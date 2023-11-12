
from assets.src import DCT
import numpy as np

qY = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
]

qC = [
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
]

def quatizeY(pic, quantization_matrix):
    p = np.zeros_like(pic)
    for i in range(8):
        for j in range(8):
            p[i][j] = round(pic[i][j] / quantization_matrix[i][j])
    return p

def quatizeUV(pic, quantization_matrix):
    p = np.zeros_like(pic)
    for i in range(8):
        for j in range(8):
            p[i][j] = round(pic[i][j] / quantization_matrix[i][j])
    return p

def iQuatizeY(pic, quantization_matrix):
    p = np.zeros_like(pic)
    for i in range(8):
        for j in range(8):
            p[i][j] = pic[i][j] * quantization_matrix[i][j]
    return p

def iQuatizeUV(pic, quantization_matrix):
    p = np.zeros_like(pic)
    for i in range(8):
        for j in range(8):
            p[i][j] = pic[i][j] * quantization_matrix[i][j]
    return p
