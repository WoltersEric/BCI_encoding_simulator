import math
import numpy as np
import copy
occurence_list = [2, 7, 7, 5, 1, 1, 2, 1]

def retrieve_distr(occurence_list):
    solved = False
    j = 1
    factor = 2
    nessary = [2]
    for i in occurence_list[1:]:
        nessary.extend([nessary[-1] * factor + i])
    while solved == False:
        available = np.power(2, range(j, len(occurence_list)+j))
        for i, (x, y) in enumerate(zip(available, nessary)):
            if x < y:
                for z in range(1, i+1):
                    if nessary[i-z] <= 1:
                        if i == 1:
                            j += 1
                            break
                        else:
                            pass
                    else:
                        nessary[i-z] -= 1
                        for k in range(z):
                            nessary[i-z+k+1] = nessary[i-z] * factor + occurence_list[i-z+k+1]
                        break
                break
            if i == len(nessary)-1:
                solved = True
                break
    return available, nessary, j

def create_ham_list(hamming_d):
    dicts = {}
    keys = range(hamming_d, hamming_d+10)
    if hamming_d == 2:
        opt1_codes = ['11', '00']
        opt2_codes = ['10', '01']
    else:
        raise Exception("This hammming distance is not yet implemented")
    for i in keys:
        temp1_codes = []
        temp2_codes = []
        for (code1, code2) in zip(opt1_codes, opt2_codes):
            if len(opt1_codes[0]) < len(code1):
                break
            opt1_codes.extend([code1 + '0', code2 + '1'])
            temp1_codes.extend([code1])
            opt2_codes.extend([code1 + '1', code2 + '0'])
            temp2_codes.extend([code2])
        for (code1, code2) in zip(temp1_codes, temp2_codes):
            opt1_codes.remove(code1)
            opt2_codes.remove(code2)
        dicts.update({i: copy.deepcopy(opt1_codes)})
    return dicts

if __name__ == "__main__":
    # test, nesar, j = retrieve_distr(occurence_list)
    test = create_ham_list(2)