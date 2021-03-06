import heapq
import numpy as np
import operator

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

    def make_heap(self, frequency):
        for key in frequency:
            node = HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    @staticmethod
    def determine_avg_length(codes, frequency):
        indv_len = [len(codes[k]) * frequency[k] for k in codes.keys()]
        return sum(indv_len)

    @staticmethod
    def merge_nodes(heap):
        count = 0
        while (len(heap) > 1):
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)

            merged = HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(heap, merged)
            count += 1
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

        self.make_VLEC_codes_helper(root.left, current_code + "01")
        self.make_VLEC_codes_helper(root.right, current_code + "10")

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
        node = HeapNode("leaf", 0, count=len(frequency))
        heapq.heappush(self.min_que, node)
        for i in range(len(frequency)-1):
            node = heapq.heappop(self.min_que)
            counter = node.count - 1
            node1 = HeapNode("leaf", node.freq + self.left, count=counter)
            node2 = HeapNode("leaf", node.freq + self.right, count=counter)
            heapq.heappush(self.min_que, node1)
            heapq.heappush(self.min_que, node2)
            self.min_que.sort(key=operator.attrgetter('count'), reverse=True)
            self.min_que.sort(key=operator.attrgetter('freq'), reverse=False)
        for i in range(len(frequency)):
            node3 = self.min_que[i]
            key = list(frequency)[i]
            tot = HeapNode(key, node3.count, frequency[key], node3.freq)
            heapq.heappush(self.heap, tot)

    def merge_min_nodes(self):
        while (len(self.heap) > 1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)
            if (round(node1.count, 5) == round(node2.count, 5)) and (round(self.left-self.right, 5) != 0):
                node3 = heapq.heappop(self.heap)
                node1, node3 = node3, node1
                heapq.heappush(self.heap, node3)
            elif round(abs(node1.count-node2.count), 5) != round(abs(self.left - self.right), 5):
                if node1.count > node2.count is self.left < self.right:
                    node3 = heapq.heappop(self.heap)
                    node1, node3 = node3, node1
                    heapq.heappush(self.heap, node3)
                else:
                    node3 = heapq.heappop(self.heap)
                    node2, node3 = node3, node2
                    heapq.heappush(self.heap, node3)
            if (self.left < self.right) is (node1.count > node2.count):
                node1, node2 = node2, node1
            freq = node1.freq + 1
            merged = HeapNode(None, freq, node1.cost + node2.cost, (node1.count - self.left))

            merged.left = node1
            merged.right = node2
            heapq.heappush(self.heap, merged)
            self.heap.sort(key=operator.attrgetter('count'), reverse=True)
            self.heap.sort(key=operator.attrgetter('freq'), reverse=False)

    def make_weighted_codes_helper(self, root, current_code):
        if (root == None):
            return

        if (root.char != None):
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            self.temp.append([root.char, root.cost, root.count])
            return

        if (root.left.count >= root.right.count) and (root.left.cost >= root.right.cost):
            root.left, root.right = root.right, root.left

        if self.left <= self.right:
            low_cost = '0'
            high_cost = '1'
        else:
            low_cost = '1'
            high_cost = '0'
        self.make_weighted_codes_helper(root.left, current_code + low_cost)
        self.make_weighted_codes_helper(root.right, current_code + high_cost)

    def make_weighted_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_weighted_codes_helper(root, current_code)

    def create_code(self, frequency):
        if self.right >= self.left:
            self.left = self.left / self.right
            self.right = 1
        else:
            self.right = self.right / self.left
            self.left = 1

        frequency = {k: frequency[k] for k in sorted(frequency, key=frequency.get, reverse=True)}
        self.min_que = []
        self.temp = []
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}
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
        dimensions = self.determine_rectangle(length)
        q = 1
        count = 0
        first_time = True
        while True:
            base = np.identity(q, dtype=int).tolist()
            codes = [k + [1] for k in base]
            if q > dimensions[0]-1:
                if first_time:
                    codes = codes[:(dimensions[0]-1-q)]
                    first_time = False
                else:
                    codes = codes[-(dimensions[0]-q):(dimensions[0]-1-q)]
            for current_code in codes:
                self.codes[list(frequency.keys())[count]] = ''.join(map(str, current_code))
                count += 1
                if count == length:
                    return self.codes
            q += 1

    @staticmethod
    def determine_rectangle(number):
        answer = 1
        while (answer ** 2) < number:
            answer += 1
        dimensions = [answer, ((number-1)//answer+1)]
        return dimensions

