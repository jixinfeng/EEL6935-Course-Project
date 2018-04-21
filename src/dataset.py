import tensorflow as tf
import numpy as np
from types import SimpleNamespace
import os

def bow_parser(vocab_size, is_multiclass=True):
    """
        Returns function for parsing sparse bag-of-words text files and encoding feature vectors
        for either binomial or multinomial regression.
    """
    # Produces y vectors that look like [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    # Produces x vectors that look like [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, ...]
    def encode_one_hot_vectors(line):
        tokens = line.split()
        class_, bow = (int(tokens[0]) - 1), tokens[1:]

        x = np.zeros(vocab_size, np.int8)

        # Mark every word appearing in the document (movie review)
        for word in bow:
            index, count = word.split(b':')
            index = int(index) - 1
            x[index] = 1

        if is_multiclass:
            y = np.zeros(10, np.int8)
            # Mark which class the document belongs to
            y[class_] = 1
        else:
            y = np.zeros(2, np.int8)
            # Threshold the document class
            y[0 if class_ <= 5 else 1] = 1
        return x, y

    return lambda line: tf.py_func(encode_one_hot_vectors, [line], [tf.int8, tf.int8])

def preview_dataset(input_function, num_items=5):
    """
    Create a new tf.Session and preview the first num_items entires of the dataset
    :param num_items: the number of items to preview
    :param input_function: a function creating dataset to preview
    :return: void
    """
    dataset = input_function()
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        for i in range(num_items):
            print(sess.run(next_element))


def numpy_dataset(filename):
    """
    :param filename: the numpy file to load
    :return: a function that constructs a tensorflow dataset from the numpy file
    """
    with np.load(filename) as data:
        inputs = data['inputs']
        labels = data['labels']
    return SimpleNamespace(
        input_fn=lambda: tf.data.Dataset.from_tensor_slices((inputs, labels)),
        size=inputs.shape[0]
    )


def parsed_text_dataset(parser, files, num_parallel_calls=1):
    # Map the movie ratings (strings) to an array index
    dataset = tf.data.TextLineDataset(files)
    dataset = dataset.map(parser, num_parallel_calls=num_parallel_calls)
    return dataset


def bow_dataset(bow_files, is_multiclass=True):
    vocab_file = os.path.join(os.path.dirname(__file__), 'data/aclImdb/imdb.vocab')

    with open(vocab_file) as f:
        vocab_size = sum(1 for _ in f.readlines())
    assert vocab_size == 89527, "unexpected vocab size %d for imdb dataset." % vocab_size

    return SimpleNamespace(
        input_fn=lambda: parsed_text_dataset(bow_parser(vocab_size, is_multiclass=is_multiclass), bow_files),
        size=25000 # hardcoded for now...
    )