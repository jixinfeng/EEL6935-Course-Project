import numpy as np
import tensorflow as tf

from model import LogisticRegression

VOCAB_SIZE = 89526
TRAINING_BOW = "aclImdb/train/labeledBow.feat"

def process_data(vocab_size, class_ix):

    # Produces y vectors that look like [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    # Produces x vectors that look like [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, ...]
    def encode_one_hot_vectors(line):
        tokens = line.split()
        class_, bow = tokens[0], tokens[1:]

        x = np.zeros(vocab_size, np.int8)
        y = np.zeros(len(class_ix.keys()), np.int8)

        # Mark every word appearing in the document (movie review)
        for word in bow:
            index, count = word.split(':')
            index = int(index) - 1
            x[index] = 1

        # Mark which class the document belongs to
        y[class_ix[class_]] = 1

        return x, y

    return lambda line: tf.py_func(encode_one_hot_vectors, [line], [tf.int8, tf.int8])


def get_dataset():
    # Map the movie ratings (strings) to an array index
    class_ix = {str(rating): rating - 1 for rating in range(1,11)}
    dataset = tf.data.TextLineDataset(TRAINING_BOW)
    dataset = dataset.map(process_data(VOCAB_SIZE, class_ix), num_parallel_calls=1)
    return dataset


def main():
    # Create a logistic regression model,
    # and train on the Stanford Movie Review Dataset
    model = LogisticRegression()
    model.train(get_dataset())


if __name__ == "__main__":
    main()