from code_generation import HuffmanCoding, RowColumn, WeightedHuffmanCoding, VLECHuffmanCoding
from language_model import Rnn
import os
import numpy as np
import re
import matplotlib.pyplot as plt
"""
Test text -> code words -> noise added -> faulty words replaced by backspace and correct letter -> repeat untill no faults -> validate
"""
#TODO: fix backspace option. what is the probability?
#TODO: what happens between prob/char2indice missmatch in language model

class PerformanceEstimation:
    def __init__(self, path, coding_algorithm, noise=[0.6, 0.68]):
        self.path = path
        self.coding_algorithm = coding_algorithm
        self.noise = noise   # , 0.99]  # [x] or [fp, fn]
        self.codes = None
        self._test_text_data = self._get_test_data(self.path)
        self.chars = list(set(self._test_text_data))
        self.chars.append("bck")
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.codes_to_freq = None
        self.coding = None
        self.RNN = Rnn()
        self.initial_freq = self.determine_frequencies(None)
        self.breakout = False
        self.prior = []

    def _get_test_data(self, path):
        with open(self.path, 'r+') as file:
            text = file.read()
            # text = re.sub('[^a-z ]+', '', text)
        test_text_data = text.rstrip()

        return test_text_data

    def determine_frequencies(self, prior):
        prob = self.RNN.predict_letter_prob(prior)
        prob = {k: prob[k] for k in prob.keys() if k in self.chars}
        prob.update({"bck": 0.1})
        return prob

    def init_sim(self):
        if self.coding_algorithm == "Huffman":
            self.coding = HuffmanCoding()
        elif self.coding_algorithm == "RowColumn":
            self.coding = RowColumn()
        elif self.coding_algorithm == "VLEC":
            self.coding = VLECHuffmanCoding()
        elif self.coding_algorithm == "Weight":
            self.coding = WeightedHuffmanCoding(self.noise[0], self.noise[1])
        else:
            raise TypeError("Unknown coding algorithm")
        self.codes = self.coding.create_code(self.initial_freq)
        self.codes_to_freq = {self.codes[k]: self.initial_freq[k] for k in self.codes.keys()}
        self.prior = []

    def create_coded_letter(self, character):
        encoded_text = []
        if self.prior:
            if self.coding_algorithm == "RowColumn":
                pass
            else:
                frequencies = self.determine_frequencies(self.prior)
                self.codes = self.coding.create_code(frequencies)
                self.codes_to_freq = {self.codes[k]: self.initial_freq[k] for k in self.codes.keys()}
        self.prior.extend(character)
        coded_char = self.codes[character]
        encoded_text.append([int(i) for i in coded_char])
        return encoded_text

    def determine_faulty_letter(self, letter, noisy_letter):
        count = 0
        temp_letter = ''
        max_len = max(list(self.codes.values()), key=len)
        if letter == noisy_letter:
            return noisy_letter
        else:
            while len(temp_letter) <= len(max_len):
                if len(temp_letter) < len(noisy_letter):
                    bit = noisy_letter[count]
                    count += 1
                else:
                    bit = np.random.choice([0, 1], p=[self.noise[0], 1 - self.noise[0]]) ^ 0
                temp_letter += str(bit)
                if temp_letter in list(self.codes.values()):
                    return [int(i) for i in temp_letter]
            return [int(i) for i in temp_letter]

    @staticmethod
    def hamming1(str1, str2):
        return sum(c1 != c2 for c1, c2 in zip(str1, str2))

    def determine_VLEC_letter(self, letter, noisy_letter):
        count = 0
        max_len = max(list(self.codes.values()), key=len)
        temp_letter = ''
        if letter == noisy_letter:
            return noisy_letter
        else:
            while len(temp_letter) <= len(max_len):
                match_values = []
                if len(temp_letter) < len(letter):
                    bit = noisy_letter[count]
                    count += 1
                else:
                    bit = np.random.choice([0, 1], p=[self.noise[0], 1 - self.noise[0]]) ^ 0
                temp_letter += str(bit)
                if len(temp_letter) >= len(letter):
                    if len(temp_letter) % 2 == 0:
                        matches = [x for x in list(self.codes.values()) if len(x) == len(temp_letter)]
                        # match_values = [match for match in matches if self.hamming1(temp_letter, match) <= 1]
                        match_values = [self.hamming1(temp_letter, match) for match in matches]
                        test = [match for match in match_values if match == 0]
                        if test:
                            temp_letter = matches[0]
                            return [int(i) for i in temp_letter]
                        test = [match for match in match_values if match == 1]
                        if test:
                            if len(test) == 1:
                                temp_letter = matches[0]
                                return [int(i) for i in temp_letter]
                            else:
                                values = [self.codes_to_freq[match] for match in matches]
                                temp_letter = matches[np.argmax(np.array(values))]
                                return [int(i) for i in temp_letter]
                        test = [match for match in match_values if match > 1]
                        if test:
                            temp_letter = matches[0]
                            return [int(i) for i in temp_letter]

                    # if match_values:
                    #     if len(match_values) == 1:
                    #         temp_letter = match_values[0]
                    #         return [int(i) for i in temp_letter]
                    #     else:
                    #         values = [self.codes_to_freq[match] for match in matches]
                    #         temp_letter = matches[np.argmax(np.array(values))]
                    #         return [int(i) for i in temp_letter]
            return [int(i) for i in temp_letter]

    def random_choice(self, bit):
        if bit == 1:
            p = self.noise[1]
        else:
            p = self.noise[0]
        random_choice = np.random.choice([0, 1], size=None, p=[p, 1 - p])
        return random_choice

    def add_noise(self, encoded_text):
        noisy_coded_text = []
        for character in encoded_text:
            if len(self.noise) > 1:
                noise_vector = [bit*self.random_choice(bit) for bit in character]
            else:
                noise_vector = np.random.choice([0, 1], size=len(character), p=[self.noise[0], 1-self.noise[0]]).tolist()
            noisy_coded_text.append((np.array(character) ^ np.array(noise_vector)).tolist())
        return noisy_coded_text

    def determine_error_correction(self, char):
        error = True
        num_of_decisions = 0
        backspace = [int(i) for i in self.codes["bck"]]
        new_text = [backspace, char]
        while error:
            noisy_new_text = self.add_noise(new_text)
            for letter, noisy_letter in zip(new_text, noisy_new_text):
                if self.coding_algorithm == "VLEC":
                    if len(letter) != len(noisy_letter):
                        print('wow')
                        pass
                    noisy_letter = self.determine_VLEC_letter(letter, noisy_letter)
                else:
                    noisy_letter = self.determine_faulty_letter(letter, noisy_letter)
                if letter != noisy_letter:
                    if num_of_decisions > 500:
                        self.breakout = True
                        # print('breakout')
                        error = False
                        break
                    elif letter == backspace:
                        new_text.insert(0, backspace)
                        num_of_decisions += len(noisy_letter)
                        break
                    else:
                        new_text = [backspace, letter]
                        num_of_decisions += len(noisy_letter)
                        break
                    break
                else:
                    num_of_decisions += len(letter)
                    if letter != backspace:
                        error = False
                    else:
                        if len(new_text) > 1:
                            del new_text[0]
                            del noisy_new_text[0]
        return num_of_decisions

    def simulate(self):
        self.breakout = False
        self.init_sim()
        best_num_of_decisions = 0
        num_of_decisions = 0
        for character in self._test_text_data:
            if character not in self.initial_freq.keys():
                continue
            else:
                encoded_letter = self.create_coded_letter(character)
                noisy_coded_letter = self.add_noise(encoded_letter)[0]
                best_num_of_decisions += len(encoded_letter[0])
                if self.breakout == False:
                    if self.coding_algorithm == "VLEC":
                        noisy_coded_letter = self.determine_VLEC_letter(encoded_letter[0], noisy_coded_letter)
                    else:
                        noisy_coded_letter = self.determine_faulty_letter(encoded_letter[0], noisy_coded_letter)
                    if encoded_letter[0] != noisy_coded_letter:
                        num_of_decisions += len(noisy_coded_letter)
                        num_of_decisions += self.determine_error_correction(encoded_letter[0])
                    else:
                        num_of_decisions += len(encoded_letter)
                else:
                    return [best_num_of_decisions, 0]
        return [best_num_of_decisions, num_of_decisions]


def visualize_results():
    interval = 2.6
    result_optimal = {
        'RowColumn': [],
        'Huffman': [],
        'VLEC': [],
        # 'Weighted': [],
    }
    temp = {
        'RowColumn': [],
        'Huffman': [],
        'VLEC': [],
        # 'Weighted': [],
    }
    result_actual = {
        'RowColumn': [],
        'Huffman': [],
        'VLEC': [],
        # 'Weighted': [],
    }
    x_axis = list(np.arange(0.75, 0.99, 0.025))
    algorithms = ["Huffman", "RowColumn", "VLEC"]
    for i in x_axis:
        temp['Huffman'] = []
        temp['RowColumn'] = []
        temp['VLEC'] = []
        # temp['Huffman'] = []
        for j in range(5):
            for algor in algorithms:
                Algorithm = PerformanceEstimation("text.txt", algor, noise=[i])
                tmp = Algorithm.simulate()
                if tmp[1] == 0:
                    pass
                else:
                    temp[algor].append(tmp[1])
        for algor in algorithms:
            average = 0
            if len(temp[algor]) > 0:
                average = sum(temp[algor]) / len(temp[algor])
            result_actual[algor].append(average * interval / 60)
        print("stap klaar")
    for algor in algorithms:
        Algorithm = PerformanceEstimation("text.txt", algor, noise=[0.9999])
        tmp = Algorithm.simulate()
        result_optimal[algor].append(tmp[0] * interval / 60)
    return [x_axis, result_optimal, result_actual]

def plot(results):
    x_axis = results[0]
    result_optimal = results[1]
    result_actual = results[2]
    fig, ax = plt.subplots()
    horiz_line_datah = np.array([result_optimal['Huffman']for i in x_axis])
    ax.plot(x_axis, horiz_line_datah, color='red', alpha=0.4)
    y_axish =np.array(result_actual['Huffman'])
    y_axish[y_axish == 0] = 'nan'
    ax.plot(x_axis, y_axish, color='red', alpha=0.7, label="Huffman paradigm")

    horiz_line_datar = np.array([result_optimal['RowColumn'] for i in x_axis])
    ax.plot(x_axis, horiz_line_datar, color='blue', alpha=0.4)
    y_axisr =np.array(result_actual['RowColumn'])
    y_axisr[y_axisr == 0] = 'nan'
    ax.plot(x_axis, y_axisr, color='blue', alpha=0.7, label="RowColumn paradigm")

    horiz_line_datav = np.array([result_optimal['VLEC'] for i in x_axis])
    ax.plot(x_axis, horiz_line_datav, color='black', alpha=0.4)
    y_axisv =np.array(result_actual['VLEC'])
    y_axisv[y_axisv == 0] = 'nan'
    ax.plot(x_axis, y_axisv, color='black', alpha=0.7, label="VLEC paradigm")
    # horiz_line_data = np.array([result_optimal['Weighted'] for i in x_axis])
    # ax.plot(x_axis, horiz_line_data, color='red', alpha=0.4)
    # ax.plot(x_axis, result_actual['Weighted'], color='green', alpha=0.7, label="Weighted clicks")
    ax.set_xlabel('Click accuracy')
    ax.set_ylabel('Time in minutes')
    ax.set_title('Performance of different spelling paradigms')
    ax.legend()
    plt.show()

results = visualize_results()
plot(results)
# Test1 = PerformanceEstimation("text.txt", "Huffman")
# Test2 = PerformanceEstimation("text.txt", "RowColumn")
# Test3 = PerformanceEstimation("text.txt", "VLEC")
# best_num_of_decisions_huffman, num_of_decisions_huffman = Test3.simulate()
# best_num_of_decisions_rowcolumn, num_of_decisions_rowcolumn = Test2.simulate()