from assets.src.DCT import *
import matplotlib.pyplot as plt

image = cv2.imread('../image/lena.jpg', cv2.IMREAD_GRAYSCALE)
result = dct(image)
reconstruction = idct(result)
# cv2.imshow('original',image)
# cv2.imshow('reconstruction',reconstruction)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


plt.gray()
plt.subplot(121), plt.imshow(image), plt.axis('off'), plt.title('original image', size=20)
plt.subplot(122), plt.imshow(reconstruction), plt.axis('off'), plt.title('reconstructed image (DCT+IDCT)', size=20)
plt.show()