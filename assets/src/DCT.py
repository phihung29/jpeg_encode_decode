import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# dct block 8x8
def dct(array):
    result = np.zeros_like(array)
    # dct row
    for i in range(8):
        for u in range(8):
            sum=0
            cu = 1/np.sqrt(2) if u==0 else 1
            for v in range(8):
                sum += array[i][v]*np.cos((2 * v + 1) * np.pi * u / 16)
            result[i][u]=sum*cu*1/2
    #dct col
    for j in range(8):
        for u in range(8):
            sum=0
            cu = 1/np.sqrt(2) if u==0 else 1
            for v in range(8):
                sum += array[v][j]*np.cos((2 * v + 1) * np.pi * u / 16)
            result[u][j]=sum*cu*1/2
    return result
# idct block 8x8
def idct(result):
    reconstruction = np.zeros_like(result)
    #idct row
    for i in range(8):
        for v in range(8):
            sum=0
            for u in range(8):
                cu=1/np.sqrt(2) if u ==0 else 1
                sum+=result[i][u]*cu*np.cos((2*v+1)*np.pi*u/16)
            sum*=1/2
            reconstruction[i][v]=sum + 128
    #idct col
    for j in range(8):
        for v in range(8):
            sum=0
            for u in range(8):
                cu=1/np.sqrt(2) if u ==0 else 1
                sum+=result[u][j]*cu*np.cos((2*v+1)*np.pi*u/16)
            sum*=1/2
            reconstruction[v][j]=sum+128
    return reconstruction

#dct function
def dct_image(image):
    height, width = image.shape
    block_size = 8
    blocks_w = width + (block_size - width % block_size) if width%block_size != 0 else width
    blocks_h = height + (block_size - height % block_size) if height%block_size != 0 else height

    new_image = np.zeros((blocks_h, blocks_w))
    new_image[:height, :width] = image

    new_image = new_image.astype(float)

    result = np.zeros_like(new_image)

    for i in range(0, blocks_h, block_size):
        for j in range(0, blocks_w, block_size):
            block = new_image[i:i+block_size,j:j+block_size]
            result[i:i+block_size,j:j+block_size]=dct(block)
    return result

#idct function
def idct_image(result):
    height, width = result.shape
    block_size = 8

    image = np.zeros((height, width))

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block=result[i:i+block_size,j:j+block_size]
            image[i:i+block_size,j:j+block_size]=idct(block)
    return image


# image = cv2.imread('../image/lena.jpg')
# image_original = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# YCbCr_image = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
#
# Y,Cb,Cr = cv2.split(YCbCr_image)
#
# new = cv2.merge([Y,Cb,Cr])
#
#
#
#
# new_image = cv2.cvtColor(YCbCr_image,cv2.COLOR_YCrCb2RGB)
#
#
#
# plt.subplot(121), plt.imshow(image_original), plt.axis('off'), plt.title('Original Image', size=10)
#
# # Hiển thị kết quả DCT trước khi lượng tử hóa
# plt.subplot(122), plt.imshow(new_image), plt.axis('off'), plt.title('DCT Result', size=10)
# plt.show()