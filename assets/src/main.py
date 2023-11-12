from assets.src import quantization
from assets.src.DCT import *
import matplotlib.pyplot as plt

image = cv2.imread('../image/lena.jpg', cv2.IMREAD_GRAYSCALE)
result = dct(image)

# Original and Reconstructed images (for comparison)
original_image = image.copy()

# ảnh ban đầu
plt.gray()
plt.subplot(231), plt.imshow(original_image), plt.axis('off'), plt.title('Original Image', size=10)

# Hiển thị kết quả DCT trước khi lượng tử hóa
plt.subplot(232), plt.imshow(result), plt.axis('off'), plt.title('DCT Result', size=10)

# Lượng tử hóa
quantization_factor = 1
quantized_result_Y = quantization.quatizeY(result, quantization_factor * quantization.qY)
quantized_result_UV = quantization.quatizeUV(result, quantization_factor * quantization.qC)

# Hiển thị kết quả DCT sau khi lượng tử hóa
plt.subplot(233), plt.imshow(quantized_result_Y), plt.axis('off'), plt.title('Quantized DCT Result', size=10)

# Áp dụng lượng tử hóa ngược cho thành phần Y và UV để khôi phục giá trị ban đầu
dequantized_result_Y = quantization.iQuatizeY(quantized_result_Y, quantization_factor * quantization.qY)
dequantized_result_UV = quantization.iQuatizeUV(quantized_result_UV, quantization_factor * quantization.qC)

# Thực hiện phép biến đổi ngược của DCT để tái tạo thành phần Y và UV
reconstructed_Y = idct(dequantized_result_Y)
reconstructed_UV = idct(dequantized_result_UV)

# Hiển thị ảnh tái tạo
plt.subplot(234), plt.imshow(reconstructed_Y), plt.axis('off'), plt.title('Reconstructed Y', size=10)
plt.subplot(235), plt.imshow(reconstructed_UV), plt.axis('off'), plt.title('Reconstructed UV', size=10)

# Show the plots
plt.show()