from assets.src import quantization
from assets.src.DCT import *
import matplotlib.pyplot as plt

def main() -> object:
    # Load and preprocess the image
    image = cv2.imread('../image/lena.jpg', cv2.IMREAD_GRAYSCALE)
    result = dct(image)

    # Original and Reconstructed images (for comparison)
    original_image = image.copy()

    # Display the original image
    plt.gray()
    plt.subplot(231), plt.imshow(original_image), plt.axis('off'), plt.title('Original Image', size=10)

    # Display the DCT result before quantization
    plt.subplot(232), plt.imshow(result), plt.axis('off'), plt.title('DCT Result', size=10)

    # Quantization
    quantization_factor = 1
    quantized_result_Y = quantization.quantizeY(result, quantization_factor * quantization.qY)
    quantized_result_UV = quantization.quantizeUV(result, quantization_factor * quantization.qC)

    # Display the quantized DCT result
    plt.subplot(233), plt.imshow(quantized_result_Y), plt.axis('off'), plt.title('Quantized DCT Result', size=10)

    # Dequantize the Y and UV components
    dequantized_result_Y = quantization.iQuantizeY(quantized_result_Y, quantization_factor * quantization.qY)
    dequantized_result_UV = quantization.iQuantizeY(quantized_result_UV, quantization_factor * quantization.qC)

    # Inverse DCT to reconstruct Y and UV components
    reconstructed_Y = idct(dequantized_result_Y)
    reconstructed_UV = idct(dequantized_result_UV)

    # Display the reconstructed Y and UV components
    plt.subplot(234), plt.imshow(reconstructed_Y), plt.axis('off'), plt.title('Reconstructed Y', size=10)
    plt.subplot(235), plt.imshow(reconstructed_UV), plt.axis('off'), plt.title('Reconstructed UV', size=10)


    # Combine the Y and UV components to obtain the final reconstructed image
    reconstructed_image = cv2.merge([reconstructed_Y, reconstructed_UV, reconstructed_UV])

    # Convert YCrCb to RGB
    reconstructed_image_rgb = cv2.cvtColor(reconstructed_image.astype(np.uint8), cv2.COLOR_YCrCb2RGB)

    # Display the final reconstructed image
    plt.subplot(236), plt.imshow(reconstructed_image_rgb), plt.axis('off'), plt.title('Reconstructed Image', size=10)

    # Show the plots
    plt.show()

if __name__ == "__main__":
    main()
