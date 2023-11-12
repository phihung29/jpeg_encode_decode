import cv2
import numpy as np
from PIL import Image

def dct(image):
    height, width = image.shape
    block_size = 8
    blocks_w = width + (block_size - width % block_size)
    blocks_h = height + (block_size - height % block_size)

    new_image = np.zeros((blocks_h, blocks_w))
    new_image[:height, :width] = image

    new_image = new_image.astype(float)
    new_image -= 128

    result = np.zeros_like(new_image)

    for i in range(0, blocks_h, block_size):
        for j in range(0, blocks_w, block_size):
            for u in range(i, i + block_size):
                for v in range(j, j + block_size):
                    cu = 1 / np.sqrt(2) if u % 8 == 0 else 1
                    cv = 1 / np.sqrt(2) if v % 8 == 0 else 1
                    sum = 0
                    for x in range(block_size):
                        for y in range(block_size):
                            sum += new_image[x + i, y + j] * np.cos((2 * x + 1) * (u % 8) * np.pi / 16) * np.cos(
                                (2 * y + 1) * (v % 8) * np.pi / 16)
                    result[u, v] = sum * cu * cv * (1 / 4)

    return result

def idct(result):
    height, width = result.shape
    block_size = 8

    image = np.zeros((height, width))

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            for x in range(i, i + block_size):
                for y in range(j, j + block_size):
                    sum = 0
                    for u in range(block_size):
                        for v in range(block_size):
                            cu = 1 / np.sqrt(2) if u == 0 else 1
                            cv = 1 / np.sqrt(2) if v == 0 else 1
                            sum += result[u + i, v + j] * cu * cv * np.cos((2 * (x % 8) + 1) * u * np.pi / 16) * np.cos(
                                (2 * (y % 8) + 1) * v * np.pi / 16)
                    image[x, y] = sum * (1 / 4) + 128

    return image

