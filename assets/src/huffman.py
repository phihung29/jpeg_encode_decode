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


def zig_zag(matrix):
    result = [0] * (block_size * block_size)
    index = 0
    
    for i in range(block_size + block_size - 1):
        if i % 2 == 0:  # Diagonal going up
            row = min(i, block_size - 1)
            col = max(0, i - block_size + 1)
            while row >= 0 and col < block_size:
                result[index] = matrix[row][col]
                index += 1
                row -= 1
                col += 1
        else:  # Diagonal going down
            col = min(i, block_size - 1)
            row = max(0, i - block_size + 1)
            while col >= 0 and row < block_size:
                result[index] = matrix[row][col]
                index += 1
                col -= 1
                row += 1

    return result

def zig_zag_reverse(array):
    result = [[0] * block_size for _ in range(block_size)]
    index = 0
    
    for i in range(block_size + block_size - 1):
        if i % 2 == 0:  # Diagonal going up
            row = min(i, block_size - 1)
            col = max(0, i - block_size + 1)
            while row >= 0 and col < block_size:
                result[row][col] = array[index]
                index += 1
                row -= 1
                col += 1
        else:  # Diagonal going down
            col = min(i, block_size - 1)
            row = max(0, i - block_size + 1)
            while col >= 0 and row < block_size:
                result[row][col] = array[index]
                index += 1
                col -= 1
                row += 1
    return result

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

    print(zig_zag_reverse(decodedStringDC))
    # Function call
    decodedStringRLC = decode_file(probsRLC[0], encodedStringRLC)
    print("\nDecoded Huffman Data:")
    print(decodedStringRLC)

