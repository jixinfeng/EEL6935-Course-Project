from itertools import chain
import random
import os
import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict
from itertools import count
import re
import argparse

neg_folder = os.path.join(os.path.dirname(__file__), 'aclImdb/train/neg')
pos_folder = os.path.join(os.path.dirname(__file__), 'aclImdb/train/pos')

word_ix = count(1)
word_dict = defaultdict(lambda: next(word_ix))

# Visualize review lengths
    #import matplotlib.pyplot as plt
    #pos_lengths = [len(x) for x in pos_reviews]
    #neg_lengths = [len(x) for x in neg_reviews]

    #plt.hist(pos_lengths, alpha=0.5, bins=100)
    #plt.hist(neg_lengths, alpha=0.5, bins=100)
    #plt.show()


def create_fixed_len_multiclass_dataset(review_sets, max_len=200):
    total_reviews = sum([set_size for set_size, _ in review_sets])
    print("Total reviews:", total_reviews)
    inputs = np.zeros([total_reviews, max_len])
    labels = np.zeros([total_reviews, 10])
    print("Allocated memory")
    i = 0
    for _, review_generator in review_sets:
        for score, review in review_generator():
            truncated = review[:max_len]
            inputs[i, :len(truncated)] = list(map(lambda x: word_dict[x], truncated))
            labels[i, score-1] = 1
            i += 1
            if i % 100 == 0:
                print(i, "/", total_reviews)
    print("Finished")
    return inputs, labels


def load_reviews_folder(folder, num_reviews=None):
    filenames = os.listdir(folder)
    if num_reviews != 0:
        assert num_reviews <= len(filenames)
        filenames = random.sample(filenames, num_reviews)

    def get_next_review():
        for filename in filenames:
            score = int(re.search('.*_([0-9]+)\.txt', filename).group(1))
            with open(os.path.join(folder, filename)) as f:
                review = word_tokenize(f.read().lower())
            yield score, review

    return len(filenames), get_next_review


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate numpy file of IMDB review sequences')
    parser.add_argument('-num-reviews', type=int, default=0, help='0 to use all reviews')
    parser.add_argument('-max-len', type=int, default=200, help='Max number of words per review. ' +
                                                                'Excess words will be truncated')
    parser.add_argument('-o', "--output_file", type=str, default='output', help='output dataset location')

    args = parser.parse_args()
    pos_reviews = load_reviews_folder(pos_folder, num_reviews=int(args.num_reviews/2))
    neg_reviews = load_reviews_folder(neg_folder, num_reviews=int(args.num_reviews/2))

    inputs, labels = create_fixed_len_multiclass_dataset([pos_reviews, neg_reviews], max_len=args.max_len)

    print("Total vocab size: ", len(word_dict.keys()))

    np.savez(args.output_file, inputs=inputs, labels=labels)
    np.save(args.output_file+"_dict", dict(word_dict))

