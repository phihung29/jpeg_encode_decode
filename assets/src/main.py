from assets.src import quantization
from assets.src.DCT import *
import matplotlib.pyplot as plt

image = cv2.imread('../image/lena.jpg', cv2.IMREAD_GRAYSCALE)
result = dct(image)
quantized_result_Y = quantization.quatizeY(result, quantization.qY)
quantized_result_UV = quantization.quatizeUV(result, quantization.qC)
# (Nếu cần) Thực hiện ngược lại lượng tử hóa (đưa về giá trị ban đầu) cho thành phần Y
# dequantized_result_Y = quantization.iQuatizeY(quantized_result_Y, quantization.qY)
# (Nếu cần) Thực hiện ngược lại lượng tử hóa (đưa về giá trị ban đầu) cho thành phần UV
# dequantized_result_UV = quantization.iQuatizeUV(quantized_result_UV, quantization.qC)
reconstruction = idct(result)
# cv2.imshow('original',image)
# cv2.imshow('reconstruction',reconstruction)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


plt.gray()
plt.subplot(121), plt.imshow(image), plt.axis('off'), plt.title('original image', size=20)

# plt.subplot(122), plt.imshow(reconstruction), plt.axis('off'), plt.title('reconstructed image (DCT+IDCT)', size=20)
plt.show()