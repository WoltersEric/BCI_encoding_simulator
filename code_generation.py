import heapq
import numpy as np
import copy
import operator

"""
frequency input is dict frequency {a} = x
"""
# frequency = {
#     "A": 40,
#     "B": 20,
#     "C": 10,
#     "D": 6,
#     "E": 5,
#     "F": 5,
#     "G": 4,
#     "H": 3,
#     "I": 3,
#     "J": 3,
#     "K": 1}

class HeapNode:
    def __init__(self, char, freq, cost=None, count=None):
        self.char = char
        self.freq = freq
        self.cost = cost
        self.count = count
        self.left = None
        self.right = None

    # defining comparators less_than and equals
    def __lt__(self, other):
        return self.freq < other.freq

    def __eq__(self, other):
        if (other == None):
            return False
        if (not isinstance(other, HeapNode)):
            return False
        return self.freq == other.freq

class HuffmanCoding:
    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}
    # functions for compression:

    def make_heap(self, frequency):
        for key in frequency:
            node = HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    @staticmethod
    def merge_nodes(heap):
        while (len(heap) > 1):
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)

            merged = HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(heap, merged)
        return heap

    def make_codes_helper(self, root, current_code):
        if (root == None):
            return

        if (root.char != None):
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)

    def create_code(self, frequency):
        self.make_heap(frequency)
        self.heap = self.merge_nodes(self.heap)
        self.make_codes()
        return self.codes

class VLECHuffmanCoding(HuffmanCoding):
    def __init__(self):
        super().__init__()

    def make_VLEC_codes_helper(self, root, current_code):
        if (root == None):
            return

        if (root.char != None):
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return

        self.make_codes_helper(root.left, current_code + "010")
        self.make_codes_helper(root.right, current_code + "101")

    def make_VLEC_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_VLEC_codes_helper(root, current_code)

    def create_code(self, frequency):
        self.make_heap(frequency)
        self.heap = self.merge_nodes(self.heap)
        self.make_VLEC_codes()
        return self.codes


class WeightedHuffmanCoding(HuffmanCoding):
    def __init__(self, left, right):
        self.min_que = []
        self.left = left
        self.right = right
        self.temp = []
        super().__init__()

    def make_min_que(self, frequency):
        node = HeapNode("leaf", 0, count=0)
        heapq.heappush(self.min_que, node)
        for i in range(len(frequency)-1):
            node = heapq.heappop(self.min_que)
            node1 = HeapNode("leaf", node.freq + self.left, count=len(frequency)-i)
            node2 = HeapNode("leaf", node.freq + self.right, count=len(frequency)-i)
            heapq.heappush(self.min_que, node1)
            heapq.heappush(self.min_que, node2)
        self.min_que.sort(key=operator.attrgetter('count'), reverse=True)
        self.min_que.sort(key=operator.attrgetter('freq'), reverse=False)
        for i in range(len(frequency)):
            # test.append(self.min_que[i])
            node3 = self.min_que[i]
            key = list(frequency)[i]
            tot = HeapNode(key, node3.count, frequency[key], node3.freq)
            heapq.heappush(self.heap, tot)
        self.heap.sort(key=operator.attrgetter('freq'), reverse=False)

    def merge_min_nodes(self):
        while (len(self.heap) > 1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)
            freq = node1.freq + 1
            countt = (node1.count - self.left + node2.count - self.right)
            merged = HeapNode(None, freq, node1.cost + node2.cost, countt)

            merged.left = node1
            merged.right = node2
            heapq.heappush(self.heap, merged)

    def make_weighted_codes_helper(self, root, current_code):
        if (root == None):
            return

        if (root.char != None):
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            self.temp.append([root.char, root.cost, root.count])
            return

        if (root.left.count > root.right.count) and (root.left.cost > root.right.cost):
            root.left, root.right = root.right, root.left

        self.make_weighted_codes_helper(root.left, current_code + "0")
        self.make_weighted_codes_helper(root.right, current_code + "1")

    def make_weighted_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_weighted_codes_helper(root, current_code)

    def create_code(self, frequency):
        frequency = {k: frequency[k] for k in sorted(frequency, key=frequency.get, reverse=True)}
        self.make_min_que(frequency)
        self.merge_min_nodes()
        self.make_weighted_codes()
        return self.codes

class RowColumn():
    def __init__(self):
        self.codes = {}

    def create_code(self, frequency):
        length = len(list(frequency.keys()))
        frequency = {k: frequency[k] for k in sorted(frequency, key=frequency.get, reverse=True)}
        q = 1
        count = 0
        while True:
            base = np.identity(q, dtype=int).tolist()
            codes = [k + [1] for k in base]
            for current_code in codes:
                self.codes[list(frequency.keys())[count]] = ''.join(map(str, current_code))
                count += 1
                if count == length:
                    return self.codes
            q += 1

# h = HuffmanCoding()
# codes = h.create_code(frequency)
# bla = WeightedHuffmanCoding(3, 1)
# weighted_codes = bla.create_code(frequency)
# vlec = VLECHuffmanCoding()
# vlec_codes = vlec.create_code(frequency)
# test = RowColumn()
# codes2 = test.create_code(frequency)