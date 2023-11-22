import heapq
import collections
import numpy as np

block_size = 8

class MinHeapNode:
    def __init__(self, data, count):
        self.left = None
        self.right = None
        self.data = data
        self.count = count

    def __lt__(self, other):
        return self.count < other.count


def storeCodes(root, str1,codes):
    if root is None:
        return
    if root.data != '$':
        codes[root.data] = str1
    storeCodes(root.left, str1 + "0",codes)
    storeCodes(root.right, str1 + "1",codes)


def HuffmanCodes(minHeap,size,codes):

    heapq.heapify(minHeap)

    while len(minHeap) != 1:
        left = heapq.heappop(minHeap)
        right = heapq.heappop(minHeap)
        top = MinHeapNode('$', left.count + right.count)
        top.left = left
        top.right = right
        heapq.heappush(minHeap, top)
    storeCodes(minHeap[0], "",codes)


def decode_file(root, s):
    ans = []
    curr = root
    n = len(s)
    for i in range(n):
        if s[i] == '0':
            curr = curr.left
        else:
            curr = curr.right

        # reached leaf node
        if curr.left is None and curr.right is None:

            ans.append(curr.data)
            # reset
            curr = root
    return ans

##  Zig-zag
def zig_zag(input_matrix):
    z = np.empty([block_size*block_size])
    index = -1
    bound = 0
    for i in range(0, 2 * block_size - 1):
        if i < block_size:
            bound = 0
        else:
            bound = i - block_size + 1
        for j in range(bound, i - bound + 1):
            index += 1
            if i % 2 == 1:
                z[index] = input_matrix[j, i-j]
            else:
                z[index] = input_matrix[i-j, j]
    return z

# def zig_zag_reverse(input_matrix):
#     output_matrix = np.empty([block_size, block_size])
#     index = -1
#     bound = 0
#     input_m = []
#     for i in range(0, 2 * block_size - 1):
#         if i < block_size:
#             bound = 0
#         else:
#             bound = i - block_size + 1
#         for j in range(bound, i - bound + 1):
#             index += 1
#             if i % 2 == 1:
#                 output_matrix[j, i - j] = input_matrix[0][index]
#             else:
#                 output_matrix[i - j, j] = input_matrix[0][index]
#     return output_matrix

# Driver code
if __name__ == "__main__":
    # xu ly block

    qtable = np.array([[2, 0, 3, 0, 0, 0, 0, 0,],
                       [0, 0, 4, 0, 0, 0, 0, 0,],
                       [0, 5, 0, 0, 0, 0, 0, 0,],
                       [0, 0, 6, 0, 0, 0, 0, 0,],
                       [0, 0, 0, 0, 0, 0, 0, 0,],
                       [0, 0, 0, 0, 0, 0, 0, 0,],
                       [0, 0, 0, 0, 0, 0, 0, 0,],
                       [0, 0, 0, 0, 0, 0, 0, 0,]])
    dc = zig_zag(qtable)
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
    HuffmanCodes(probsDPCM,len(dc),codeDC)
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
    if(zeros != 0):
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
    HuffmanCodes(probsRLC,len(rlc1),codeRLC)
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

    # print(zig_zag_reverse(decodedStringDC))
    # Function call
    decodedStringRLC = decode_file(probsRLC[0], encodedStringRLC)
    print("\nDecoded Huffman Data:")
    print(decodedStringRLC)

