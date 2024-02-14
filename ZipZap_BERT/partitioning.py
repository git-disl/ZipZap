import numpy as np
import sys
sys.path.append("..")
import pickle as pkl

n_bucket = 8

with open("./data/vocab", "rb") as f:
    vocab = pkl.load(f)

total_freq = np.sum(vocab.frequency)
print(len(vocab.frequency))
unit_freq = total_freq/n_bucket

offset = 0
bucket_list = []

for i in range(n_bucket):
    lower = offset
    count = 0
    for j in range(lower, len(vocab.frequency)):
        count += vocab.frequency[j]
        if count >= unit_freq or j == len(vocab.frequency)-1:
            upper = j
            break

    bucket_list.append([lower, upper])
    offset = upper + 1

print(bucket_list)






