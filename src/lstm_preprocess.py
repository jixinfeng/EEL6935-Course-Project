from itertools import chain
import random
import os
import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict
from itertools import count
import re
import argparse

neg_folder = None
pos_folder = None

# Visualize review lengths
    #import matplotlib.pyplot as plt
    #pos_lengths = [len(x) for x in pos_reviews]
    #neg_lengths = [len(x) for x in neg_reviews]

    #plt.hist(pos_lengths, alpha=0.5, bins=100)
    #plt.hist(neg_lengths, alpha=0.5, bins=100)
    #plt.show()


def get_unknown_dict(review_sets, max_len=200, threshold=1):
    word_counts = defaultdict(int)
    for _, review_generator in review_sets:
        for score, review in review_generator():
            truncated = review[:max_len]
            for word in truncated:
                word_counts[word.lower()] += 1

    ok_counts = filter(lambda w_c: w_c[1] > threshold, word_counts.items())
    # word no 1 is the "unknown" token
    word_dict = defaultdict(lambda: 1, {w: i for i, (w, c) in enumerate(ok_counts)})
    return word_dict


def create_fixed_len_multiclass_dataset(review_sets, max_len=200, word_dict=None):
    total_reviews = sum([set_size for set_size, _ in review_sets])
    print("Total reviews:", total_reviews)
    inputs = np.zeros([total_reviews, max_len])
    labels = np.zeros([total_reviews, 10])
    print("Allocated memory")
    i = 0

    if word_dict is None:
        word_dict = get_unknown_dict(review_sets, max_len=max_len, threshold=1)
        save_dict = dict(word_dict)
    else:
        save_dict = word_dict

    print("Vocab size:", len(word_dict.keys()))

    for _, review_generator in review_sets:
        for score, review in review_generator():
            truncated = review[:max_len]
            inputs[i, :len(truncated)] = list(map(lambda x: word_dict[x.lower()], truncated))
            labels[i, score-1] = 1
            i += 1
            if i % 500 == 0:
                print(("  %5d/%5d \r" % (i, total_reviews)), end="")

    print("\nFinished")
    return inputs, labels, save_dict


def load_reviews_folder(folder, num_reviews=0):
    filenames = os.listdir(folder)
    if num_reviews != 0:
        assert num_reviews <= len(filenames)
        filenames = random.sample(filenames, num_reviews)

    def get_next_review():
        for filename in filenames:
            score = int(re.search('.*_([0-9]+)\.txt', filename).group(1))
            with open(os.path.join(folder, filename)) as f:
                review = word_tokenize(f.read().lower())
                #review = word_tokenize(f.read().replace('/', ' / ').replace('-', ' - ').replace('.', ' . '))
            yield score, review

    return len(filenames), get_next_review


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate numpy file of IMDB review sequences')
    parser.add_argument('type', type=str, help='train | test')
    parser.add_argument('-num-reviews', type=int, default=0, help='0 to use all reviews')
    parser.add_argument('-max-len', type=int, default=200, help='Max number of words per review. ' +
                                                                'Excess words will be truncated')
    parser.add_argument('-o', "--output_file", type=str, default='output', help='output dataset location')
    parser.add_argument('-dict', type=str, default=None, help='dictionary to use')

    args = parser.parse_args()

    assert args.type in ['train', 'test'], 'argument "type" must be either "train" or "test"'

    if args.dict:
        word_dict = defaultdict(lambda: 1, np.load(args.dict).item())
    else:
        word_dict = None

    neg_folder = os.path.join(os.path.dirname(__file__), 'data/aclImdb/'+args.type+'/neg')
    pos_folder = os.path.join(os.path.dirname(__file__), 'data/aclImdb/'+args.type+'/pos')

    pos_reviews = load_reviews_folder(pos_folder, num_reviews=int(args.num_reviews/2))
    neg_reviews = load_reviews_folder(neg_folder, num_reviews=int(args.num_reviews/2))

    inputs, labels, word_dict = create_fixed_len_multiclass_dataset([pos_reviews, neg_reviews],
                                                                        max_len=args.max_len,
                                                                        word_dict=word_dict)

    np.savez(args.output_file, inputs=inputs, labels=labels)
    np.save(args.output_file+"_dict", dict(word_dict))

