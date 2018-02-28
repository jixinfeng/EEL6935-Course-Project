import numpy as np
from collections import Counter
import tensorflow as tf

VOCAB_SIZE = 89526
TRAINING_BOW = "aclImdb/train/labeledBow.feat"


class LogisticRegression:
    def __init__(self):
        self.width_in = None
        self.width_out = None

    # Returns a tensorflow variable containing the loss function
    def _build_model(self):
        with tf.name_scope('fully_connected'):
            x = tf.placeholder(tf.float32, [None, self.width_in])
            y = tf.placeholder(tf.float32, [None, self.width_out])

            # Set model weights
            W = tf.Variable(tf.zeros([self.width_in, self.width_out]))
            b = tf.Variable(tf.zeros([self.width_in]))

        with tf.name_scope('softmax'):
            # Construct model
            pred = tf.nn.softmax(tf.matmul(x, W) + b)

        with tf.name_scope('loss'):
            # Minimize error using cross entropy
            return tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

    def train(self, dataset, num_epochs=25, learning_rate=0.01, batch_size=2):
        # Network depth, # of output classes
        self.width_in = train_x.shape()[1]
        self.width_out = train_y.shape()[1]

        # Create the tensorflow variables used
        loss = self._build_model()

        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Prepare the dataset batches
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()

        # Start training
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(train_x / batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    # Fit training using batch data
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                                  y: batch_ys})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if (epoch + 1) % display_step == 0:
                    print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost)

            print "Optimization Finished!"


def one_hot_encoder(vocab_size, class_ix):

    def encode(line):
        tokens = line.values[0].split()
        class_, bow = tokens[0], tokens[1:]

        x = np.zeros(vocab_size, np.int8)
        y = np.zeros(len(class_ix.keys()), np.int8)

        for word in bow:
            index, count = [int(x) for x in word.split(':')]
            x[index] = 1

        y[class_ix[class_]]

        return x, y

    return lambda line: tf.py_func(encode, [line], [tf.int8, tf.int8])


def main():
    class_ix = {rating: int(rating) for rating in range(1,10)}

    dataset = tf.data.TextLineDataset(TRAINING_BOW)
    dataset = dataset.map(one_hot_encoder(VOCAB_SIZE, class_ix), num_parallel_calls=2)

    model = LogisticRegression()
    model.train()


if __name__ == "__main__":
    main()