from code_generation import HuffmanCoding, RowColumn, WeightedHuffmanCoding, VLECHuffmanCoding
from language_model import Rnn
import os
import numpy as np
import math
import copy
import re
import matplotlib.pyplot as plt
import scipy.stats
import operator
"""
Test text -> code words -> noise added -> faulty words replaced by backspace and correct letter -> repeat untill no faults -> validate
"""
#TODO: what happens between prob/char2indice missmatch in language model

class PerformanceEstimation:
    def __init__(self, path, coding_algorithm, noise=[0.6, 0.68], iterations=1):
        self.path = path
        self.coding_algorithm = coding_algorithm
        self.noise = noise   # , 0.99]  # [x] or [fp, fn]
        self.codes = None
        self.interval = 2.6
        self._test_text_data = self._get_test_data(self.path)
        self.chars = list(set(self._test_text_data))
        self.chars.append("bck")
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.codes_to_freq = None
        self.coding = None
        self._get_code_algorithm()
        self.RNN = Rnn()
        self.initial_freq = self.determine_frequencies(None)
        self.breakout = False
        self.prior = []
        self.iterations = iterations

    def _get_test_data(self, path):
        with open(self.path, 'r+') as file:
            text = file.read()
            # text = re.sub('[^a-z ]+', '', text)
        test_text_data = text.rstrip()

        return test_text_data

    def _get_code_algorithm(self):
        if self.coding_algorithm == "Huffman" or self.coding_algorithm == "Huffman2":
            self.coding = HuffmanCoding()
        elif self.coding_algorithm == "RowColumn":
            self.coding = RowColumn()
        elif self.coding_algorithm == "VLEC":
            self.coding = VLECHuffmanCoding()
        elif self.coding_algorithm == "Weighted":
            self.coding = WeightedHuffmanCoding(self.noise[1], self.noise[0])
        else:
            raise TypeError("Unknown coding algorithm")

    def determine_frequencies(self, prior):
        prob = self.RNN.predict_letter_prob(prior)
        prob = {k: prob[k] for k in prob.keys() if k in self.chars}
        prob.update({"bck": 0.1})
        return prob

    def init_sim(self):
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
                    bit = np.random.choice([0, 1], p=[max(self.noise), 1 - max(self.noise)]) ^ 0
                temp_letter += str(bit)
                if temp_letter in list(self.codes.values()):
                    return [int(i) for i in temp_letter]
            return [int(i) for i in temp_letter]

    def determine_faulty_letter_test(self, letter, noisy_letter):
        count = 0
        temp_letter = ''
        possible_codes = list(self.codes.values())
        possible_len = [len(code) for code in possible_codes]
        max_len = max(possible_codes, key=len)
        max_noise = max(self.noise)
        additional_des = 0
        if letter == noisy_letter:
            return [noisy_letter, 0]
        else:
            while len(temp_letter) <= len(max_len):
                if len(temp_letter) < len(noisy_letter):
                    bit = noisy_letter[count]
                    count += 1
                else:
                    bit = np.random.choice([0, 1], p=[max(self.noise), 1 - max(self.noise)]) ^ 0
                temp_letter += str(bit)
                if len(temp_letter) in possible_len:
                    additional_des += 1
                    decision = np.random.choice([0, 1], p=[1-max_noise, max_noise])
                    if len(temp_letter) == letter:
                        if decision == 1:
                            return [self.determine_closest_match(temp_letter, letter, possible_codes), additional_des]
                        else:
                            pass
                    else:
                        if decision == 1:
                            pass
                        else:
                            return [self.determine_closest_match(temp_letter, letter, possible_codes), additional_des]
        return [[int(i) for i in temp_letter], additional_des]
    
    # def binairy_search(self):
    #     # use language model to retreive probability distribution for indv letters
    #     # divide group up in two equal selections
    #     # if group is selected multiply by true positive/negative rate
    #     # redivede until one option is equal to other options aka > p = 0.5
    #     # select corrosponding letter

    def determine_closest_match(self, temp_letter, letter, possible_codes):
        opt = min([self.hamming1(temp_letter, code) for code in possible_codes if
                   len(code) == len(temp_letter)])
        matches = [code for code in possible_codes if
                      len(code) == len(temp_letter) and self.hamming1(temp_letter, code) == opt]
        if len(matches) > 1:
            values = [self.codes_to_freq[match] for match in matches]
            temp_letter = matches[np.argmax(np.array(values))]
            return [int(i) for i in temp_letter]
        else:
            return [int(i) for i in matches[0]]

    @staticmethod
    def hamming1(str1, str2):
        return sum(c1 != c2 for c1, c2 in zip(str1, str2))

    def determine_VLEC_letter(self, letter, noisy_letter):
        count = 0
        possible_codes = list(self.codes.values())
        possible_codes = sorted(possible_codes, key=len)
        max_len = max(possible_codes, key=len)
        temp_letter = ''
        if letter == noisy_letter:
            return noisy_letter
        else:
            while len(temp_letter) < len(max_len):
                if len(temp_letter) < len(letter):
                    bit = noisy_letter[count]
                    count += 1
                else:
                    bit = np.random.choice([0, 1], p=[self.noise[0], 1 - self.noise[0]]) ^ 0
                temp_letter += str(bit)
                possible_codes = [e for e in possible_codes if len(e) >= len(temp_letter)]
                if len(temp_letter) == len(possible_codes[0]):
                    match_dist = np.array([[code, self.hamming1(temp_letter, code[:len(possible_codes[0])])] for code in possible_codes])
                    if match_dist[match_dist[:, 1] == '0'].size != 0:
                        if match_dist[match_dist[:, 1] == '0'].shape[0] == 1:
                            temp_letter = match_dist[match_dist[:, 1] == '0'][0, 0]
                            return [int(i) for i in temp_letter]
                        else:
                            pass
                    elif match_dist[match_dist[:, 1] == '1'].size != 0:
                        temp_arr = match_dist[match_dist[:, 1] == '1']
                        short, long = [], []
                        for i in range(temp_arr.shape[0]):
                            if len(temp_arr[i, 0]) == len(temp_arr[0, 0]):
                                short.append(self.codes_to_freq[temp_arr[i, 0]])
                            else:
                                long.append(self.codes_to_freq[temp_arr[i, 0]])
                        if max(short) >= sum(long):
                            temp_letter = [key for (key, value) in self.codes_to_freq.items() if value == max(short)][0]
                            return [int(i) for i in temp_letter]
                        else:
                            pass
                    else:
                        opt = min([self.hamming1(temp_letter, match) for match in possible_codes if self.hamming1(temp_letter, match) > 1])
                        temp_letter = match_dist[match_dist[:, 1] == str(opt)][0, 0]
                        return [int(i) for i in temp_letter]
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
                noise_vector = [self.random_choice(bit) for bit in character]
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
                    noisy_letter = self.determine_VLEC_letter(letter, noisy_letter)
                elif self.coding_algorithm == "Huffman2":
                    noisy_letter, addition_num = self.determine_faulty_letter_test(letter, noisy_letter)
                    num_of_decisions += addition_num
                else:
                    noisy_letter = self.determine_faulty_letter(letter, noisy_letter)
                if len(noisy_letter) < 1:
                    raise Exception('dit kan niet. The value of x was: {}'.format(noisy_letter))
                if letter != noisy_letter:
                    if num_of_decisions > 500:
                        # self.breakout = True
                        # print('breakout')
                        return num_of_decisions
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

    def simulate(self, noise):
        self.noise = noise
        if self.coding_algorithm == "Weighted":
            self.coding = WeightedHuffmanCoding(self.noise[1], self.noise[0])
        optimal_result = []
        real_result = []
        for i in range(self.iterations):
            self.init_sim()
            self.breakout = False
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
                        elif self.coding_algorithm == "Huffman2":
                            noisy_letter, addition_num = self.determine_faulty_letter_test(encoded_letter[0], noisy_coded_letter)
                            num_of_decisions += addition_num
                        else:
                            noisy_coded_letter = self.determine_faulty_letter(encoded_letter[0], noisy_coded_letter)
                        if encoded_letter[0] != noisy_coded_letter:
                            num_of_decisions += len(noisy_coded_letter)
                            num_of_decisions += self.determine_error_correction(encoded_letter[0])
                        else:
                            num_of_decisions += len(encoded_letter[0])
                    elif self.breakout == True:
                        num_of_decisions = num_of_decisions
                        pass
            if num_of_decisions != 0:
                optimal_result.append(best_num_of_decisions * self.interval / 60)
                real_result.append(num_of_decisions * self.interval / 60)
        if len(optimal_result) > self.iterations/2:
            real_result = [value for value in real_result if value != 0]
            average = sum(real_result) / len(real_result)
        else:
            optimal_result = [0]
            average = 0
        return [max(optimal_result), average]
        # return [max(optimal_result), real_result]


def visualize_results():
    interval = 2.6
    biased_weight = False
    result_optimal = {
        'RowColumn': [],
        'Huffman': [],
        'Huffman2': [],
        'VLEC': [],
        'Weighted': [],
    }
    result_actual = {
        'RowColumn': [],
        'Huffman': [],
        'Huffman2': [],
        'VLEC': [],
        'Weighted': [],
    }
    x_axis = list(np.arange(0.75, 0.99, 0.025))
    algorithms = ["Huffman", "RowColumn", "Weighted"]
    if biased_weight:
        for algor in algorithms:
            Algorithm = PerformanceEstimation("text.txt", algor, iterations=25)
            for count, i in enumerate(x_axis):
                tmp_list = []
                for j in x_axis:
                    tmp = Algorithm.simulate(noise=[i, j])
                    tmp_list.append(tmp[1])
                    if (count == len(x_axis) - 1) and (j > 0.97):
                        result_optimal[algor].append(tmp[0])
                result_actual[algor].append(tmp_list)
                print('stap {0}/{1} van {2} klaar'.format(count+1, len(x_axis), algor))
    else:
        for algor in algorithms:
            Algorithm = PerformanceEstimation("text.txt", algor, iterations=3)
            for count, i in enumerate(x_axis):
                tmp = Algorithm.simulate(noise=[i, 0.8])
                result_actual[algor].append(tmp[1])
                if count == len(x_axis) - 1:
                    result_optimal[algor].append(tmp[0])
                print('stap {0}/{1} van {2} klaar'.format(count+1, len(x_axis), algor))
    return [x_axis, result_optimal, result_actual]


def error_bar(data):
    temp = []
    for i in data:
        a = 1.0 * np.array(i)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        temp.append([m, m-h])
    return temp

def plot(results):
    x_axis = results[0]
    x_axis = [round(x, 2) for x in x_axis]
    result_optimal = results[1]
    result_actual = results[2]
    biased_weight = False
    algorithms = ["Huffman", "RowColumn", "Weighted"]
    if biased_weight:
        for algor in algorithms:
            result = copy.deepcopy(np.array(result_actual[algor]))
            result.astype(float)
            fig, ax = plt.subplots()
            im = ax.imshow(result, cmap='RdBu_r', vmin=0, vmax=90)
            ax.set_xticks(np.arange(len(x_axis)))
            ax.set_yticks(np.arange(len(x_axis)))
            ax.set_xticklabels(x_axis)
            ax.set_yticklabels(x_axis)
            ax.set_title('Performance of {} spelling paradigm in min for 25 iterations'.format(algor))
            ax.set_xlabel('True negative rate')
            ax.set_ylabel('True positive rate')
            fig.colorbar(im, ax=ax)
            fig.set_label('Time to spell full sentence in minutes')
            fig.tight_layout()
            ax.legend()
            plt.show()
    else:
        fig, ax = plt.subplots()
        horiz_line_datah = np.array([result_optimal['Huffman']for i in x_axis])
        ax.plot(x_axis, horiz_line_datah, color='red', alpha=0.4)
        y_axish = np.array(result_actual['Huffman'])
        y_axish[y_axish == 0] = 'nan'
        ax.plot(x_axis, y_axish, color='red', alpha=0.7, label="Huffman paradigm")

        # horiz_line_datah = np.array([result_optimal['Huffman2']for i in x_axis])
        # ax.plot(x_axis, horiz_line_datah, color='yellow', alpha=0.4)
        # y_axish = np.array(result_actual['Huffman2'])
        # y_axish[y_axish == 0] = 'nan'
        # ax.plot(x_axis, y_axish, color='yellow', alpha=0.7, label="Huffman with error check paradigm")

        horiz_line_datar = np.array([result_optimal['RowColumn'] for i in x_axis])
        ax.plot(x_axis, horiz_line_datar, color='blue', alpha=0.4)
        y_axisr = np.array(result_actual['RowColumn'])
        y_axisr[y_axisr == 0] = 'nan'
        ax.plot(x_axis, y_axisr, color='blue', alpha=0.7, label="RowColumn paradigm")
        #
        # horiz_line_datav = np.array([result_optimal['VLEC'] for i in x_axis])
        # ax.plot(x_axis, horiz_line_datav, color='black', alpha=0.4)
        # y_axisv = np.array(result_actual['VLEC'])
        # y_axisv[y_axisv == 0] = 'nan'
        # ax.plot(x_axis, y_axisv, color='black', alpha=0.7, label="VLEC paradigm")
        #
        horiz_line_datav = np.array([result_optimal['Weighted'] for i in x_axis])
        ax.plot(x_axis, horiz_line_datav, color='green', alpha=0.4)
        y_axisv = np.array(result_actual['Weighted'])
        y_axisv[y_axisv == 0] = 'nan'
        ax.plot(x_axis, y_axisv, color='green', alpha=0.7, label="Weighted paradigm")
        ax.set_xlabel('Click accuracy')
        ax.set_ylabel('Time in minutes')
        ax.set_title('Performance of different spelling paradigms with biased noise [x, 0.8] for 30 iterations')
        ax.legend()
        plt.show()

results = visualize_results()
plot(results)