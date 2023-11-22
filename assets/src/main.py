import collections

from assets.src import quantization
from assets.src.DCT import *
from assets.src.huffman import *
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
    quantization_factor = 10
    quantized_result_Y = quantization.quantizeY(result, quantization_factor * quantization.qY)
    quantized_result_UV = quantization.quantizeUV(result, quantization_factor * quantization.qC)

    # Display the quantized DCT result
    plt.subplot(233), plt.imshow(quantized_result_Y), plt.axis('off'), plt.title('Quantized DCT Result', size=10)

    # Dequantize the Y and UV components
    dequantized_result_Y = quantization.iQuantizeY(quantized_result_Y, quantization_factor * quantization.qY)
    dequantized_result_UV = quantization.iQuantizeY(quantized_result_UV, quantization_factor * quantization.qC)
    print("Quantized Y block:\n", quantized_result_Y)
    print("Dequantized Y block:\n", dequantized_result_Y)

    dc = zig_zag(dequantized_result_Y)
    print("DC/n")
    print(dc)
    # Huffman DPCM
    # Tìm tần suất xuất hiện cho mỗi giá trị của danh sách
    counterDPCM = collections.Counter(dc)

    # Xác định danh sách các giá trị dưới dạng danh sách các cặp (điểm, Số lần xuất hiện tương ứng)
    probsDPCM = []
    for key, value in counterDPCM.items():
        probsDPCM.append(MinHeapNode(key, counterDPCM[value]))

    codeDC = {}
    HuffmanCodes(probsDPCM, len(dc), codeDC)
    print("Character With there Frequencies:")
    for key in sorted(codeDC):
        print(key, codeDC[key])

    encodedStringDC = ""
    decodedStringDC = []
    for i in dc:
        encodedStringDC += codeDC[i]

    print("\nEncoded Huffman data:")
    print(encodedStringDC)

    rlc1 = []
    zeros = 0
    for i in range(1, len(dc)):
        if (dc[i] == 0):
            zeros += 1
        else:
            rlc1.append(zeros)
            rlc1.append(dc[i])
            zeros = 0
    if (zeros != 0):
        rlc1.append(zeros)
        rlc1.append(0)
    print("RLC/n")
    print(rlc1)
    # rlc1 =[1, -2, 1, 2, 3 ,-1, 4,0]
    # Huffman RLC
    # Tìm tần suất xuất hiện cho mỗi giá trị của danh sách
    counterRLC = collections.Counter(rlc1)
    # Xác định danh sách giá trị dưới dạng danh sách các cặp (điểm, Số lần xuất hiện tương ứng)
    probsRLC = []
    for key, value in counterRLC.items():
        probsRLC.append(MinHeapNode(key, counterRLC[value]))

    codeRLC = {}
    HuffmanCodes(probsRLC, len(rlc1), codeRLC)
    print("\nCharacter With there Frequencies:")
    for key in sorted(codeRLC):
        print(key, codeRLC[key])

    encodedStringRLC = ""
    decodedStringRLC = []
    for i in rlc1:
        encodedStringRLC += codeRLC[i]

    print("\nEncoded Huffman data:")
    print(encodedStringRLC)

    # Function call
    decodedStringDC = decode_file(probsDPCM[0], encodedStringDC)
    print("\nDecoded Huffman Data:")
    print(decodedStringDC)
    # print(zig_zag_reverse(decodedStringDC))

    print(zig_zag_reverse(decodedStringDC))
    # Function call
    decodedStringRLC = decode_file(probsRLC[0], encodedStringRLC)
    print("\nDecoded Huffman Data:")
    print(decodedStringRLC)

    # # Inverse DCT to reconstruct Y and UV components
    # reconstructed_Y = idct(dequantized_result_Y)
    # reconstructed_UV = idct(dequantized_result_UV)


    # # Display the reconstructed Y and UV components
    # plt.subplot(234), plt.imshow(reconstructed_Y), plt.axis('off'), plt.title('Reconstructed Y', size=10)
    # plt.subplot(235), plt.imshow(reconstructed_UV), plt.axis('off'), plt.title('Reconstructed UV', size=10)
    #
    #
    # # Combine the Y and UV components to obtain the final reconstructed image
    # reconstructed_image = cv2.merge([reconstructed_Y, reconstructed_UV, reconstructed_UV])

    # # Convert YCrCb to RGB
    # reconstructed_image_rgb = cv2.cvtColor(reconstructed_image.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    #
    # # Display the final reconstructed image
    # plt.subplot(236), plt.imshow(reconstructed_image_rgb), plt.axis('off'), plt.title('Reconstructed Image', size=10)

    # Show the plots
    plt.show()

if __name__ == "__main__":
    main()
