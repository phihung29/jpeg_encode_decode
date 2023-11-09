from assets.src.DCT import *

image = cv2.imread('../image/lena.jpg', cv2.IMREAD_GRAYSCALE)
result = dct(image)
reconstruction = idct(result)
cv2.imshow('original',image)
cv2.imshow('reconstruction',reconstruction)
cv2.waitKey(0)
cv2.destroyAllWindows()