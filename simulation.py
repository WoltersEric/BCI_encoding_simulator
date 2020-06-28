from code_generation import HuffmanCoding, RowColumn, WeightedHuffmanCoding, VLECHuffmanCoding
from language_model import Rnn
import numpy as np
import string
import pandas as pd

class Simulation:
    def __init__(self, path, coding_algorithm, human_error, iterations):
        self.path = path
        self.coding_algorithm = coding_algorithm
        self.noise = [0.9, 0.8]  # [x] or [TN, TP]
        self.human_error = human_error
        self.iterations = iterations

        self.codes = None
        self.codes_to_freq = None
        self.codes_to_char = None
        self.coding = None
        self.breakout = False
        self.prior = []

        self._test_text_data = self._get_test_data()
        self.chars = list(string.ascii_lowercase)
        self.chars.append("bck")
        self.chars.append(" ")
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}

        self._get_code_algorithm()
        self.RNN = Rnn()
        self.initial_freq = self.determine_frequencies(None)

    def _get_test_data(self):
        """" Returns text of target sentence/word to be spelled """
        with open(self.path, 'r+') as file:
            text = file.read()
        test_text_data = text.rstrip()
        return test_text_data

    def _get_code_algorithm(self):
        """" Initializes coding algorithm """
        if self.coding_algorithm == "Huffman" or self.coding_algorithm == "Huffman2":
            self.coding = HuffmanCoding()
        elif self.coding_algorithm == "RowColumn":
            self.coding = RowColumn()
        elif self.coding_algorithm == "VLEC":
            self.coding = VLECHuffmanCoding()
        elif self.coding_algorithm == "Weighted":
            self.coding = WeightedHuffmanCoding(self.noise[0], self.noise[1])
        else:
            raise TypeError("Unknown coding algorithm")

    def determine_frequencies(self, prior):
        """" Returns probabilities list for current typed letters(prior) """
        prob = self.RNN.predict_letter_prob(prior)
        prob = {k: prob[k] for k in prob.keys() if k in self.chars}
        prob.update({"BS": 0.2})
        return prob

    def init_sim(self):
        """" Initializes simulation by retrieving initial frequencies and codes for every letter """
        self.codes = self.coding.create_code(self.initial_freq)
        self.codes_to_freq = {self.codes[k]: self.initial_freq[k] for k in self.codes.keys()}
        self.codes_to_char = {y: x for x, y in self.codes.items()}
        self.prior = []

    def create_coded_letter(self, character):
        """" Returns binary coded version of character """
        encoded_text = []
        if self.prior:
            if self.coding_algorithm == "RowColumn":
                pass
            else:
                frequencies = self.determine_frequencies(self.prior)
                self.codes = self.coding.create_code(frequencies)
                self.codes_to_freq = {self.codes[k]: self.initial_freq[k] for k in self.codes.keys()}
                self.codes_to_char = {y: x for x, y in self.codes.items()}
        coded_char = self.codes[character]
        encoded_text.append([int(i) for i in coded_char])
        return encoded_text

    def decode_huffman(self, noisy_letter):
        """" Decodes typed bitstring based on original bitstring with noise applied """
        count = 0
        temp_letter = ''
        possible_codes = list(self.codes.values())
        possible_codes = sorted(possible_codes, key=len)
        max_len = max(possible_codes, key=len)
        while len(temp_letter) <= len(max_len):
            if len(temp_letter) < len(noisy_letter):
                bit = noisy_letter[count]
                count += 1
            else:
                bit = self.random_choice(0) ^ 0
            temp_letter += str(bit)
            if temp_letter in list(self.codes.values()):
                return [int(i) for i in temp_letter]
        return [int(i) for i in temp_letter]

    def decode_vlec(self, noisy_letter):
        """" Decodes typed bitstring based on original bitstring with noise applied """
        count = 0
        temp_letter = ''
        possible_codes = list(self.codes.values())
        possible_codes = sorted(possible_codes, key=len)
        max_len = max(possible_codes, key=len)
        while len(temp_letter) <= len(max_len):
            if len(temp_letter) < len(noisy_letter):
                bit = noisy_letter[count]
                count += 1
            else:
                bit = self.random_choice(0) ^ 0
            temp_letter += str(bit)
            # for every 2 bits
            if len(temp_letter) % 2 == 0:
                possible_codes = [e for e in possible_codes if len(e) >= len(temp_letter)]

                # determine codes with lowest hamming distance
                opt = min([self.hamming1(temp_letter, code[:len(possible_codes[0])]) for code in possible_codes])
                matches = [code for code in possible_codes if
                           len(code) == len(temp_letter) and self.hamming1(temp_letter, code) == opt]
                if matches:
                    if len(matches) > 1:
                        temp_letter = matches[0]
                        return [int(i) for i in temp_letter]
                    else:
                        return [int(i) for i in matches[0]]
                else:
                    pass
        return [int(i) for i in temp_letter]

    def decode_rowcolumn(self, letter, noisy_letter):
        """" Decodes typed bitstring based on original bitstring with noise applied """
        count = 0 # count used to iterate over the noisy bitstring
        temp_letter = ''
        dimensions = RowColumn.determine_rectangle(len(self.chars))
        one_count = 0 # number of typed 1s; two 1s lead to the selection of a letter
        while True:
            clickmade = False
            # check whether bits match
            if letter[count] == noisy_letter[count]:
                temp_letter += str(letter[count])
                if letter[count] == 1:
                    one_count += 1
                count += 1
            else:
                # bit mismatch; types 1 instead of 0
                # Assumption is that the after an incorrect row selection, an second 1 bit is made to select the first column
                if letter[count] == 0:
                    temp_letter += str(noisy_letter[count])
                    if one_count == 0:
                        while self.random_choice(1) ^ 1 != 1:
                            temp_letter += str(0)
                        temp_letter += str(1)
                    return [int(i) for i in temp_letter]
                # bit mismatch; types 0 instead of 1
                else:
                    temp_letter += str(0)
                    if one_count == 0:
                        number_of_zeros = dimensions[1] - 1
                    else:
                        number_of_zeros = dimensions[0] - 1
                    # try to type required zeros
                    while not clickmade:
                        for i in range(number_of_zeros):
                            # if this fails
                            if self.random_choice(0) ^ 0 != 0:
                                temp_letter += str(1)
                                # if it is the first 1
                                if one_count == 0:
                                    while self.random_choice(1) ^ 1 != 1:
                                        temp_letter += str(0)
                                    temp_letter += str(1)
                                return [int(i) for i in temp_letter]
                            else:
                                temp_letter += str(0)
                        # try to type 1 after zeros
                        # if fail
                        if self.random_choice(1) ^ 1 != 1:
                            temp_letter += str(0)
                        else:
                            if one_count == 0:
                                temp_letter += str(1)
                                one_count += 1
                                noisy_letter[count] == 1
                                clickmade = True
                            else:
                                temp_letter += str(1)
                                return [int(i) for i in temp_letter]
        return [int(i) for i in temp_letter]

    def determine_typed_letter(self, letter, noisy_letter):
        """" Returns the actual typed letter when noise is added """
        if ''.join(str(x) for x in noisy_letter) in self.codes.values():
            return noisy_letter
        else:
            if self.coding_algorithm == "Huffman" or self.coding_algorithm == "Weighted":
                letter = self.decode_huffman(noisy_letter)
            elif self.coding_algorithm == "RowColumn":
                letter = self.decode_rowcolumn(letter, noisy_letter)
            elif self.coding_algorithm == "VLEC":
                letter = self.decode_vlec(noisy_letter)
        return letter

    def determine_closest_match(self, temp_letter, possible_codes):
        """"Find the closest match for VLEC"""
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

    def random_choice(self, bit):
        """ Returns a bit indicating whether a bitflip is necassary. aka whether noise affects the bit"""
        if len(self.noise) > 1:
            if bit == 1:
                p = self.noise[1]
            else:
                p = self.noise[0]
        else:
            p = self.noise[0]
        random_choice = np.random.choice([0, 1], size=None, p=[p, 1 - p])
        return random_choice

    def add_noise(self, encoded_text):
        """" Adds noise to the text """
        noisy_coded = []
        if self.human_error:
            for character in encoded_text:
                noise_vector = np.random.choice([0, 1], size=len(character),
                                                p=[1-self.human_error[self.coding_algorithm],self.human_error[self.coding_algorithm]]).tolist()
                noisy_coded.append((np.array(character) ^ np.array(noise_vector)).tolist())
            encoded_text = noisy_coded
        noisy_coded_text = []
        for character in encoded_text:
            if len(self.noise) > 1:
                noise_vector = [self.random_choice(bit) for bit in character]
            else:
                noise_vector = np.random.choice([0, 1], size=len(character), p=[self.noise[0], 1-self.noise[0]]).tolist()
            noisy_coded_text.append((np.array(character) ^ np.array(noise_vector)).tolist())
        return noisy_coded_text

    def determine_rowcolumn(self, code):
        """" Determine the character of rowcolumn encoding"""
        dimensions = RowColumn.determine_rectangle(len(self.chars))
        ones_idx = [i for i, x in enumerate(code) if x == 1]
        x = ones_idx[0] % (dimensions[1])
        y = (ones_idx[1] - (ones_idx[0]+1)) % (dimensions[0])
        char = '0'*x+'1'+'0'*y+'1'
        try:
            self.codes_to_char[''.join(str(x) for x in char)]
            return self.codes_to_char[''.join(str(x) for x in char)]
        except:
            return 'empty'

    def simulate(self, noise):
        """" Run the simulation"""
        self.noise = noise
        if self.coding_algorithm == "Weighted":
            self.coding = WeightedHuffmanCoding(self.noise[0], self.noise[1])
        data = []
        columns = ['N', 'Algorithm', 'Noise', 'TotalClicks', 'NmOfOnes', 'NmOfZeros']

       # For the number of iterations
        for i in range(self.iterations):
            self.init_sim()
            best_num_of_decisions = 0
            num_of_decisions = 0
            best_num_zeros = 0
            num_zeros = 0
            best_num_ones = 0
            num_ones = 0
            sentence_typed = False
            target_id = 0
            backspaces_needed = 0
            # For every letter in the target sentence/word
            for letter in self._test_text_data:
                encoded_letter = self.create_coded_letter(letter)
                self.prior.extend(letter)
                best_num_of_decisions += len(encoded_letter[0])
                best_num_ones += encoded_letter[0].count(1)
                best_num_zeros += encoded_letter[0].count(0)
                # determine number of decisions when no mistakes are made
            self.prior = [] # corresponds to typed sentence
            # Determine number of decisions required to type target sentence with noise
            while not sentence_typed:
                if backspaces_needed > 0:
                    target_letter = "BS"
                else:
                    target_letter = self._test_text_data[target_id]
                # encode letter to binary code
                encoded_letter = self.create_coded_letter(target_letter)
                # add noise to the binary code
                noisy_coded_letter = self.add_noise(encoded_letter)[0]
                # decode noisy bitstring
                noisy_decoded_letter = self.determine_typed_letter(encoded_letter[0], noisy_coded_letter)
                num_of_decisions += len(noisy_decoded_letter)
                num_zeros += noisy_decoded_letter.count(0)
                num_ones += noisy_decoded_letter.count(1)
                # determine character from decoded bitstring
                if self.coding_algorithm == "RowColumn":
                    character = self.determine_rowcolumn(noisy_decoded_letter)
                else:
                    character = self.codes_to_char[''.join(str(x) for x in noisy_decoded_letter)]
                if character == "BS":
                    try:
                        self.prior = self.prior[:-1]
                    except:
                        pass
                elif character == 'empty':
                    pass
                else:
                    self.prior.extend(character)

                if character != target_letter:
                    if character == "BS":
                        if backspaces_needed == 0:
                            target_id -= 1
                            if target_id < 0:
                                target_id = 0
                        else:
                            backspaces_needed -= 1
                    else:
                        backspaces_needed += 1
                else:
                    if character == "BS":
                        backspaces_needed -= 1
                    else:
                        target_id += 1
                # end simulation if sentence is typed or number of decisions passes threshold
                if target_id == len(self._test_text_data) or num_of_decisions > 1250:
                    sentence_typed = True
                    data.append([i, self.coding_algorithm, 'TN={}:TP={}'.format(self.noise[0], self.noise[1]),
                                 num_of_decisions, num_ones, num_zeros])
                    data.append([i, self.coding_algorithm, 'no noise', best_num_of_decisions, best_num_ones, best_num_zeros])
        df = pd.DataFrame(data, columns=columns)
        return df

