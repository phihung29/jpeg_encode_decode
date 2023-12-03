import collections

from assets.src import *
from assets.src.DCT import *
from assets.src.huffman import *
import matplotlib.pyplot as plt
from quantization import *
def main() -> object:
    # Load and preprocess the image
    qtable = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                       [12, 12, 14, 19, 26, 58, 60, 55],
                       [14, 13, 16, 24, 40, 57, 69, 56],
                       [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103, 77],
                       [24, 35, 55, 64, 81, 104, 113, 92],
                       [49, 64, 78, 87, 103, 121, 120, 101],
                       [72, 92, 95, 98, 112, 100, 103, 99]])

    image = cv2.imread('../image/lena.jpg', cv2.IMREAD_GRAYSCALE)




    height, width = image.shape
    block_size = 8
    blocks_w = width + (block_size - width % block_size) if width % block_size != 0 else width
    blocks_h = height + (block_size - height % block_size) if height % block_size != 0 else height

    new_image = np.zeros((blocks_h, blocks_w))
    new_image[:height, :width] = image

    new_image = new_image.astype(float)
    new_image -= 128

    result = np.zeros((blocks_h, blocks_w))

    zigZag = []
    for i in range(0, blocks_h, block_size):
        for j in range(0, blocks_w, block_size):
            block = new_image[i:i + block_size, j:j + block_size]
            result[i:i + block_size, j:j + block_size] = dct(block)

            zigZag.append(np.array(zig_zag(quantizeY(dct(block)))))
    # [-16,-1,0,2,-11,1,0,....]
    dc = []
    dc.append(zigZag[0][0])  # giữ nguyên giá trị đầu tiên
    for i in range(1, len(zigZag)):
        dc.append(zigZag[i][0] - zigZag[i - 1][0])

    # RLC cho giá trị AC
    rlc = []
    zeros = 0
    for i in range(0, len(zigZag)):
        zeros = 0
        for j in range(1, len(zigZag[i])):
            # if (zigZag[i][j] == 0):
            if np.all(zigZag[i][j] == 0):
                zeros += 1
            else:
                rlc.append(zeros)
                rlc.append(zigZag[i][j])
                zeros = 0
        if (zeros != 0):
            rlc.append(zeros)
            rlc.append(0)
    counterDPCM = collections.Counter(dc)

    # Xác định danh sách các giá trị dưới dạng danh sách các cặp (điểm, Số lần xuất hiện tương ứng)
    probsDPCM = []
    for key, value in counterDPCM.items():
        probsDPCM.append(MinHeapNode(key, counterDPCM[value]))


    codeDC = {}
    HuffmanCodes(probsDPCM, len(dc), codeDC)

    # print("Character With there Frequencies:")
    # # codeDC = sorted(codeDC);
    # for key in (codeDC):
    #     print(key, codeDC[key])

    encodedStringDC = ""
    decodedStringDC = []
    for i in dc:
        encodedStringDC += codeDC[i]

    # print("\nEncoded Huffman data:")
    # print(encodedStringDC)

    # Huffman RLC
    # Tìm tần suất xuất hiện cho mỗi giá trị của danh sách
    counterRLC = collections.Counter(rlc)
    # Xác định danh sách giá trị dưới dạng danh sách các cặp (điểm, Số lần xuất hiện tương ứng)
    probsRLC = []
    for key, value in counterRLC.items():
        probsRLC.append(MinHeapNode(key, counterRLC[value]))

    codeRLC = {}
    HuffmanCodes(probsRLC,len(rlc),codeRLC)


    print("\nCharacter With there Frequencies:")
    for key in sorted(codeRLC):
        print(key, codeRLC[key])

    encodedStringRLC = ""
    decodedStringRLC = []
    for i in rlc:
        encodedStringRLC += codeRLC[i]

    # print("\nEncoded Huffman data:")
    # print(encodedStringRLC)

    # #decode
    decodedStringDC = decode_file(probsDPCM[0], encodedStringDC)
    # print("\nDecoded DC Huffman Data:")
    # print(decodedStringDC)
    #
    decodedStringRLC = decode_file(probsRLC[0], encodedStringRLC)
    # print("\nDecoded AC Huffman Data:")
    # print(decodedStringRLC)

    # Inverse DPCM
    inverse_DPCM = []
    if decodedStringDC:
        inverse_DPCM.append(decodedStringDC[0])  # giá trị đầu tiên giữ nguyên

        for i in range(1, len(decodedStringDC)):
            inverse_DPCM.append(decodedStringDC[i] + inverse_DPCM[i - 1])
    # print("/n")
    # print(inverse_DPCM)


    # Inverse RLC
    inverse_RLC = []
    for i in range(0, len(decodedStringRLC)):
        if (i % 2 == 0):
            if (decodedStringRLC[i] != 0.0):
                if (i + 1 < len(decodedStringRLC) and decodedStringRLC[i + 1] == 0):
                    for j in range(1, int(decodedStringRLC[i])):
                        inverse_RLC.append(0.0)
                else:
                    for j in range(0, int(decodedStringRLC[i])):
                        inverse_RLC.append(0.0)
        else:
            inverse_RLC.append(decodedStringRLC[i])
    # print("/n")
    # print(inverse_RLC)

    new_img = np.empty(shape=(height, width))
    iheight = 0
    iwidth = 0
    temp = []
    temp2 = []
    for i in range(0, len(inverse_DPCM)):
        temp.append(inverse_DPCM[i])
        for j in range(0, 63):
            temp.append((inverse_RLC[j + i * 63]))
        temp2.append(temp)

        # inverse Zig-Zag và nghịch đảo Lượng tử hóa các hệ số DCT
        inverse_blockq = np.multiply(np.reshape(
            zig_zag_reverse(temp2), (8, 8)), qtable)

        # inverse DCT
        inverse_dct = idct(inverse_blockq, qtable)
        for startY in range(iheight, iheight + 8, 8):
            for startX in range(iwidth, iwidth + 8, 8):
                new_img[startY:startY + 8, startX:startX + 8] = inverse_dct
        iwidth = iwidth + 8
        if (iwidth == width):
            iwidth = 0
            iheight = iheight + 8
        temp = []
        temp2 = []
    np.place(new_img, new_img > 255, 255)
    np.place(new_img, new_img < 0, 0)



    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(new_img, cmap='gray'), plt.title('Image after decompress')
    plt.xticks([]), plt.yticks([])
    plt.show()

# Original and Reconstructed images (for comparison)
    # original_image = image.copy()

    # Display the original image
    # plt.gray()
    # plt.subplot(231), plt.imshow(original_image), plt.axis('off'), plt.title('Original Image', size=10)
    #
    # # Display the DCT result before quantization
    # plt.subplot(232), plt.imshow(result), plt.axis('off'), plt.title('DCT Result', size=10)

    # # Quantization
    # quantization_factor = 10
    # quantized_result_Y = quantization.quantizeY(result, quantization_factor * quantization.qY)
    # quantized_result_UV = quantization.quantizeUV(result, quantization_factor * quantization.qC)

    # # Display the quantized DCT result
    # plt.subplot(233), plt.imshow(quantized_result_Y), plt.axis('off'), plt.title('Quantized DCT Result', size=10)

    # # Dequantize the Y and UV components
    # dequantized_result_Y = quantization.iQuantizeY(quantized_result_Y, quantization_factor * quantization.qY)
    # dequantized_result_UV = quantization.iQuantizeY(quantized_result_UV, quantization_factor * quantization.qC)
    # print("Quantized Y block:\n", quantized_result_Y)
    # print("Dequantized Y block:\n", dequantized_result_Y)

    # dc = zig_zag(dequantized_result_Y)
    # print("DC/n")
    # print(dc)
    # # Huffman DPCM
    # # Tìm tần suất xuất hiện cho mỗi giá trị của danh sách
    # counterDPCM = collections.Counter(dc)
    #
    # # Xác định danh sách các giá trị dưới dạng danh sách các cặp (điểm, Số lần xuất hiện tương ứng)
    # probsDPCM = []
    # for key, value in counterDPCM.items():
    #     probsDPCM.append(MinHeapNode(key, counterDPCM[value]))
    #
    # codeDC = {}
    # HuffmanCodes(probsDPCM, len(dc), codeDC)
    # print("Character With there Frequencies:")
    # for key in sorted(codeDC):
    #     print(key, codeDC[key])
    #
    # encodedStringDC = ""
    # decodedStringDC = []
    # for i in dc:
    #     encodedStringDC += codeDC[i]
    #
    # print("\nEncoded Huffman data:")
    # print(encodedStringDC)
    #
    # rlc1 = []
    # zeros = 0
    # for i in range(1, len(dc)):
    #     if (dc[i] == 0):
    #         zeros += 1
    #     else:
    #         rlc1.append(zeros)
    #         rlc1.append(dc[i])
    #         zeros = 0
    # if (zeros != 0):
    #     rlc1.append(zeros)
    #     rlc1.append(0)
    # print("RLC/n")
    # print(rlc1)
    # # rlc1 =[1, -2, 1, 2, 3 ,-1, 4,0]
    # # Huffman RLC
    # # Tìm tần suất xuất hiện cho mỗi giá trị của danh sách
    # counterRLC = collections.Counter(rlc1)
    # # Xác định danh sách giá trị dưới dạng danh sách các cặp (điểm, Số lần xuất hiện tương ứng)
    # probsRLC = []
    # for key, value in counterRLC.items():
    #     probsRLC.append(MinHeapNode(key, counterRLC[value]))
    #
    # codeRLC = {}
    # HuffmanCodes(probsRLC, len(rlc1), codeRLC)
    # print("\nCharacter With there Frequencies:")
    # for key in sorted(codeRLC):
    #     print(key, codeRLC[key])
    #
    # encodedStringRLC = ""
    # decodedStringRLC = []
    # for i in rlc1:
    #     encodedStringRLC += codeRLC[i]
    #
    # print("\nEncoded Huffman data:")
    # print(encodedStringRLC)
    #
    # # Function call
    # decodedStringDC = decode_file(probsDPCM[0], encodedStringDC)
    # print("\nDecoded Huffman Data:")
    # print(decodedStringDC)
    # # print(zig_zag_reverse(decodedStringDC))
    #
    # print(zig_zag_reverse(decodedStringDC))
    # # Function call
    # decodedStringRLC = decode_file(probsRLC[0], encodedStringRLC)
    # print("\nDecoded Huffman Data:")
    # print(decodedStringRLC)
    #
    # # # Inverse DCT to reconstruct Y and UV components
    # # reconstructed_Y = idct(dequantized_result_Y)
    # # reconstructed_UV = idct(dequantized_result_UV)
    #
    #
    # # # Display the reconstructed Y and UV components
    # # plt.subplot(234), plt.imshow(reconstructed_Y), plt.axis('off'), plt.title('Reconstructed Y', size=10)
    # # plt.subplot(235), plt.imshow(reconstructed_UV), plt.axis('off'), plt.title('Reconstructed UV', size=10)
    # #
    # #
    # # # Combine the Y and UV components to obtain the final reconstructed image
    # # reconstructed_image = cv2.merge([reconstructed_Y, reconstructed_UV, reconstructed_UV])
    #
    # # # Convert YCrCb to RGB
    # # reconstructed_image_rgb = cv2.cvtColor(reconstructed_image.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    # #
    # # # Display the final reconstructed image
    # # plt.subplot(236), plt.imshow(reconstructed_image_rgb), plt.axis('off'), plt.title('Reconstructed Image', size=10)
    #
    # # Show the plots
    # plt.show()

if __name__ == "__main__":
    main()
