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

def quantizeY(pic, quantization_matrix):
    p = np.zeros_like(pic, dtype=float)
    for i in range(8):
        for j in range(8):
            p[i][j] = pic[i][j] / quantization_matrix[i][j]
    return np.round(p).astype(int)

def quantizeUV(pic, quantization_matrix):
    p = np.zeros_like(pic, dtype=float)
    for i in range(8):
        for j in range(8):
            p[i][j] = pic[i][j] / quantization_matrix[i][j]
    return np.round(p).astype(int)

def iQuantizeY(pic, quantization_matrix):
    p = np.zeros_like(pic, dtype=float)
    for i in range(8):
        for j in range(8):
            p[i][j] = pic[i][j] * quantization_matrix[i][j]
    return np.round(p).astype(int)

def iQuantizeImage(img, quantization_matrix):
    iHeight, iWidth = img.shape
    result = np.zeros_like(img, dtype=int)

    for startY in range(0, iHeight, 8):
        for startX in range(0, iWidth, 8):
            block = img[startY:startY+8, startX:startX+8]

            # Bổ sung dòng và cột để làm cho kích thước thành bội số của 8x8
            block = pad_image_block(block)

            quantized_block = iQuantizeBlock(block, quantization_matrix)

            result[startY:startY+8, startX:startX+8] = quantized_block

    return result

def iQuantizeBlock(block, quantization_matrix):
    p = np.zeros_like(block, dtype=float)
    for i in range(8):
        for j in range(8):
            p[i][j] = block[i][j] * quantization_matrix[i][j]
    return np.round(p).astype(int)

def pad_image_block(block):
    block_size = 8
    rows, cols = block.shape

    # Tính toán số dòng và cột cần bổ sung
    pad_rows = block_size - rows
    pad_cols = block_size - cols

    # Bổ sung dòng và cột
    padded_block = np.pad(block, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

    return padded_block