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
        if i % 2 == 0:  # đường chéo lên
            row = min(i, block_size - 1)
            col = max(0, i - block_size + 1)
            while row >= 0 and col < block_size:
                result[index] = matrix[row][col]
                index += 1
                row -= 1
                col += 1
        else:  # đường chéo xuống
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
        if i % 2 == 0:  # đường chéo lên
            row = min(i, block_size - 1)
            col = max(0, i - block_size + 1)
            while row >= 0 and col < block_size:
                result[row][col] = array[index]
                index += 1
                row -= 1
                col += 1
        else:  # Đường chéo xuống
            col = min(i, block_size - 1)
            row = max(0, i - block_size + 1)
            while col >= 0 and row < block_size:
                result[row][col] = array[index]
                index += 1
                col -= 1
                row += 1
    return result
