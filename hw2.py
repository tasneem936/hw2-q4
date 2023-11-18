import cv2
import matplotlib.pyplot as plt
import numpy as np
import huffman
from random import random


# this function builds a list which contains the probability of appearance of each element in the given vector
def calc_prob(vector):
    index = 0
    vector = sorted(vector)
    length = len(vector)
    symbols = [vector[0]]
    res_vector = [1]
    for i in range(1, length):
        if vector[i] == vector[i - 1]:
            res_vector[index] = res_vector[index] + 1
        else:
            index = index + 1
            res_vector.insert(index, 1)
            symbols.insert(index, vector[i])

    res_vec = [i / length for i in res_vector]
    return res_vec, symbols


def entropy_length(vector):
    prob_list, no_need = calc_prob(vector)
    entropy_sum = 0
    for i in range(0, len(prob_list)):
        entropy_sum = entropy_sum - (prob_list[i] * np.log2(prob_list[i]))

    # entropy_sum = entropy_sum * len(vector)
    return entropy_sum


def huffman_code_length(vector):
    prob, symbols = calc_prob(vector)
    map_dict = list(zip(symbols, prob))
    huffman_code = huffman.codebook(map_dict)
    sum_digits = 0
    index = 0
    for i in huffman_code.values():
        sum_digits = float(sum_digits) + len(i) * prob[index]
        index += 1
    return sum_digits


img = cv2.imread(r"Clown256B.bmp")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
data = np.array(gray_img)
len_of_img = len(data)
data = data.flatten()

print("--------- section 2 -----------")
huffman_res = huffman_code_length(data) * img.shape[0] * img.shape[1]
entropy_res = entropy_length(data) * img.shape[0] * img.shape[1]
print(huffman_res)
print(entropy_res)

print("--------- section 3 -----------")

random_list = []
for i in range(len_of_img):
    random_list.append(random())

huffman_res2 = huffman_code_length(random_list) * img.shape[0] * img.shape[1]
entropy_res2 = entropy_length(random_list) * img.shape[0] * img.shape[1]
with_8digit = 8 * img.shape[0] * img.shape[1]
print(huffman_res2)
print(entropy_res2)
print(with_8digit)
