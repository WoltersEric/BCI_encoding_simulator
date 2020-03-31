from code_generation import HuffmanCoding, RowColumn, WeightedHuffmanCoding, VLECHuffmanCoding
from language_model import Rnn
import win32pipe
import win32file
import pywintypes
import sys
import os
import time

os.chdir("C:/Users/eric_/Documents/Studie/BCI_paradigm_test/")
path = "txt2.txt"

def get_test_data(path):
    with open(path, 'r+') as file:
        text = file.read()
    test_text_data = text.rstrip()
    return test_text_data


def get_max_len(codes):
    return max((len(v) for k,v in codes.items()))


def get_code_algorithm(coding_algorithm):
    if coding_algorithm == "Huffman" or coding_algorithm == "Huffman2":
        coding = HuffmanCoding()
    elif coding_algorithm == "RowColumn":
        coding = RowColumn()
    elif coding_algorithm == "VLEC":
        coding = VLECHuffmanCoding()
    elif coding_algorithm == "Weighted":
        coding = WeightedHuffmanCoding(0.86, 0.8)
    else:
        raise TypeError("Unknown coding algorithm")
    return coding


def determine_frequencies(prior, chars):
    RNN = Rnn()
    prob = RNN.predict_letter_prob(prior)
    prob = {k: prob[k] for k in prob.keys() if k in chars}
    prob.update({"BS": 0.1})
    return prob


def return_codes(algorithm, prior):
    test_text_data = get_test_data(path)
    chars = list(set(test_text_data))
    coding = get_code_algorithm(algorithm)
    frequencies = determine_frequencies(prior, chars)
    codes = coding.create_code(frequencies)
    return codes

if __name__ == "__main__":
    # start = time.time()
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # algorithm = str(sys.argv[1])
    # prior = str(sys.argv[2])
    # if prior == "@":
    #     prior = ""
    # prior = prior.replace('_', ' ')
    algorithm = "Huffman"
    prior = ""
    codes = return_codes(algorithm, prior)
    final_string = ''
    for key, value in codes.items():
        string = f'{key}:{value},'
        final_string += string
    final_string = final_string[:-1]
    print("vanhier"+final_string+"tothier")
    # end = time.time()
    # print(end - start)


    # first_run = True
    # while True:
    #     #Check pipe
    #     #If setup
    #     if first_run:
    #         test_text_data = get_test_data(path)
    #         chars = list(set(test_text_data))
    #         coding = get_code_algorithm(algorithm)
    #         RNN = Rnn()
    #         ## write to pipe, set-up complete
    #         first_run = False
    #     #if retreive codes
    #     else:
    #         prob = RNN.predict_letter_prob(prior)
    #         prob = {k: prob[k] for k in prob.keys() if k in chars}
    #         prob.update({"BS": 0.1})
    #         codes = coding.create_code(prob)
    #
    #         ## Write to pipe codes
