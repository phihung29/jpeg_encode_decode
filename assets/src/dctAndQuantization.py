
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
def dct(array, quantization_matrix):
    result = np.zeros_like(array, dtype=float)

    # DCT theo hàng
    for i in range(8):
        for u in range(8):
            cu = 1 / np.sqrt(2) if u == 0 else 1
            sum = 0
            for v in range(8):
                sum += array[i][v] * np.cos((2 * v + 1) * np.pi * u / 16)
            result[i][u] = sum * cu * 1 / 2

    # DCT theo cột
    for j in range(8):
        for u in range(8):
            cu = 1 / np.sqrt(2) if u == 0 else 1
            sum = 0
            for v in range(8):
                sum += result[v][j] * np.cos((2 * v + 1) * np.pi * u / 16)
            result[u][j] = sum * cu * 1 / 2

    # Quantization
    result = np.round(result / quantization_matrix)

    return result


# IDCT block 8x8
def idct(array, quantization_matrix):
    reconstruction = np.zeros_like(array, dtype=float)

    # IDCT theo hàng
    for i in range(8):
        for v in range(8):
            sum = 0
            for u in range(8):
                cu = 1 / np.sqrt(2) if u == 0 else 1
                sum += array[i][u] * cu * np.cos((2 * v + 1) * np.pi * u / 16)
            sum *= 1 / 2
            reconstruction[i][v] = sum

    # IDCT theo cột
    for j in range(8):
        for v in range(8):
            sum = 0
            for u in range(8):
                cu = 1 / np.sqrt(2) if u == 0 else 1
                sum += reconstruction[u][j] * cu * np.cos((2 * v + 1) * np.pi * u / 16)
            sum *= 1 / 2
            reconstruction[v][j] = sum

    # Lượng tử hóa ngược
    reconstruction = reconstruction * quantization_matrix
    return reconstruction


# DCT + quantization function
def dct_image(image, quantization_matrix):
    if len(image.shape) == 2:  # kiem tra co phai anh xam hay khong
        height, width = image.shape
        channels = 1
    else:  # anh mau
        height, width, channels = image.shape

    # so luong khoi de chia het cho 8
    block_size = 8
    blocks_w = width + (block_size - width % block_size) if width % block_size != 0 else width
    blocks_h = height + (block_size - height % block_size) if height % block_size != 0 else height

    new_image = np.zeros((blocks_h, blocks_w, channels))

    if channels == 1:
        new_image[:height, :width, 0] = image
    else:
        new_image[:height, :width, :] = image

    new_image = new_image.astype(float)
    new_image -= 128

    result = np.zeros_like(new_image)

    for c in range(channels):
        for i in range(0, blocks_h, block_size):
            for j in range(0, blocks_w, block_size):
                block = new_image[i:i + block_size, j:j + block_size, c]
                result[i:i + block_size, j:j + block_size, c] = dct(block, quantization_matrix)

    return result


# IDCT + Iquantization function for an image
def idct_image(result, quantization_matrix):
    height, width, channels = result.shape
    block_size = 8

    image = np.zeros((height, width, channels))

    for c in range(channels):
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = result[i:i + block_size, j:j + block_size, c]
                # Chuyển về giá trị pixel gốc từ 0-255
                image[i:i + block_size, j:j + block_size, c] = idct(block, quantization_matrix) + 128

    return image.clip(0, 255).astype(np.uint8)  # Đảm bảo các giá trị pixel trong khoảng 0-255
