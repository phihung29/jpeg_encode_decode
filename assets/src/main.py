import collections
from assets.src.DCT import *
from assets.src.huffman import *
import matplotlib.pyplot as plt
from quantization import *


def main() -> object:
    # Load and preprocess the image
    # original_image = cv2.imread('../image/lena.jpg')
    # image = cv2.cvtColor(original_image, cv2.COLOR_BGR2YCrCb)
    image = cv2.imread('../image/lena.jpg', cv2.IMREAD_GRAYSCALE)



    # for c in range(3):
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


    # print("\nCharacter With there Frequencies:")
    # for key in sorted(codeRLC):
    #     print(key, codeRLC[key])

    encodedStringRLC = ""
    decodedStringRLC = []
    for i in rlc:
        encodedStringRLC += codeRLC[i]

    # print("\nEncoded Huffman data:")
    # print(encodedStringRLC)


    # giai ma

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
        inverse_blockq = zig_zag_reverse(temp)


        # inverse DCT
        inverse_dct = idct(iQuantizeY(inverse_blockq))


        # for startY in range(iheight, iheight + 8):
        #     for startX in range(iwidth, iwidth + 8):
        #         new_img[startY:startY + 8, startX:startX + 8] = inverse_dct

        new_img[iheight:iheight+8,iwidth:iwidth+8] =inverse_dct
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
    cv2.imwrite("decompress.jpg", new_img)


if __name__ == "__main__":
    main()
