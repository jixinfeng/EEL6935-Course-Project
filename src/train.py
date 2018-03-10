import numpy as np
import tensorflow as tf

from model import LogisticRegression

VOCAB_SIZE = 89526
TRAINING_BOW = 'aclImdb/train/labeledBow.feat'


def process_data(vocab_size, class_type):
    class_ix = {str(rating): rating - 1 for rating in range(1,11)}

    # Produces y vectors that look like [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    # Produces x vectors that look like [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, ...]
    def encode_one_hot_vectors(line):
        tokens = line.split()
        class_, bow = tokens[0], tokens[1:]

        x = np.zeros(vocab_size, np.int8)

        # Mark every word appearing in the document (movie review)
        for word in bow:
            index, count = word.split(':')
            index = int(index) - 1
            x[index] = 1

        if class_type == 'multiclass':
            y = np.zeros(len(class_ix.keys()), np.int8)
            # Mark which class the document belongs to
            y[class_ix[class_]] = 1

        if class_type == 'binary':
            y = np.zeros(2, np.int8)
            # Mark which class the document belongs to
            y[0 if class_ix[class_] <= 5 else 1] = 1
        return x, y

    return lambda line: tf.py_func(encode_one_hot_vectors, [line], [tf.int8, tf.int8])


def preview_dataset(dataset):
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        for i in range(5):
            print(sess.run(next_element))


def input_fn(type='binary'):
    # Map the movie ratings (strings) to an array index
    dataset = tf.data.TextLineDataset(TRAINING_BOW)
    dataset = dataset.map(process_data(VOCAB_SIZE, type), num_parallel_calls=1)
    return dataset


def main():
    # Create a logistic regression model,
    # and train on the Stanford Movie Review Dataset
    model = LogisticRegression()
    preview_dataset(input_fn())
    model.train(input_fn, save_dir='/tmp/model.ckpt', num_epochs=20, batch_size=20, display_period=100,
                learning_rate=0.001)


if __name__ == '__main__':
    main()
