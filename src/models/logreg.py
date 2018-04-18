import tensorflow as tf
import datetime
import os
import numpy as np
from nltk import word_tokenize

# Built referencing https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py
# and https://web.stanford.edu/class/cs20si/2017/lectures/notes_03.pdf

class LogisticRegression:
    def __init__(self, args):
        self.width_in = None
        self.width_out = None
        self.model_dir = args.model_dir

        self.word_dict = {}
        vocab_file = os.path.join(os.path.dirname(__file__), '../data/aclImdb/imdb.vocab')
        with open(vocab_file, encoding='utf-8') as f:
            linenum = 0
            for line in f.readlines():
                self.word_dict[line[:-1]] = linenum

    def _get_widths(self, dataset):
        # Find out the input and output data dimensions
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:
            batch_xs, batch_ys = sess.run(next_element)
            # Retrieve network depth, # of output classes
            self.width_in = batch_xs.shape[0]
            self.width_out = batch_ys.shape[0]
            print("Width in: %d, width out: %d" % (self.width_in, self.width_out))

    # Returns a tensorflow variable containing the loss function
    def _build_model(self):
        with tf.name_scope('input'):
            # Input data
            self.x = tf.placeholder(tf.float32, shape=[None, self.width_in])
            self.y = tf.placeholder(tf.float32, shape=[None, self.width_out])

        with tf.name_scope('weights'):
            # Set model weights
            self.W = tf.get_variable("W", shape=(self.width_in, self.width_out))

        with tf.name_scope('biases'):
            self.b = tf.get_variable("b", shape=self.width_out)

        with tf.name_scope('logits'):
            self.logits = tf.matmul(self.x, self.W) + self.b

        with tf.name_scope('softmax'):
            preds = tf.nn.softmax(self.logits)
            self.maxpreds = tf.argmax(preds, 1)

        with tf.name_scope('loss'):
            # Minimize error using cross entropy
            loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(preds), reduction_indices=1))
            tf.summary.scalar('loss_summary', loss)
            return loss

    # Train the model with the Dataset passed in
    def train(self, input_fn, args):
        dataset = input_fn()

        # Get info on the dataset
        self._get_widths(dataset)

        # Prepare our model and acquire a reference to the loss
        loss = self._build_model()

        # Batch the data
        dataset = dataset.shuffle(buffer_size=25000)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(args.batch_size))

        # Get an iterator to the dataset
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        # Create an optimization variable
        with tf.name_scope('optimizer'):
            train_step = tf.train.AdamOptimizer(args.lr).minimize(loss)

        # Initialize all of the model variables (i.e. assign their default values)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:
            # Load the previous model if it was listed in the constructor
            saver = tf.train.Saver()
            if self.model_dir:
                saver.restore(sess, self.model_dir)
                self.model_dir = None

            # Init graph variables and set up logging
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('./logs/train/%s' % datetime.datetime.now(), sess.graph)
            sess.run(init)

            step_num = 0
            for epoch in range(args.epochs):
                sess.run(iterator.initializer)
                batch_num = 1
                running_cost = 0.0

                # Run through the epoch
                while True:
                    try:
                        # Note that each x, y pair in the batch has an epoch number
                        batch_xs, batch_ys = sess.run(next_element)

                    except tf.errors.OutOfRangeError:
                        print("End of training dataset.")
                        break

                    # Train using batch data
                    summary, c, _ = sess.run([merged, loss, train_step], feed_dict={self.x: batch_xs,
                                                                                    self.y: batch_ys})
                    # Accumulate loss
                    running_cost += c

                    # Display log message every n batches
                    if batch_num % args.display_interval == 0:
                        print("Epoch: %04d Batch: %06d cost=%.9f" % (epoch + 1, batch_num, running_cost / args.display_interval))
                        running_cost = 0.0

                    train_writer.add_summary(summary, step_num)
                    batch_num += 1
                    step_num += 1

            print("Training Finished!")
            if args.save_dir:
                save_path = saver.save(sess, args.save_dir)
                print("Saved model to "+args.save_dir)

    def score(self, input_fn, batch_size=20, display_period=100):
        tf.reset_default_graph()
        dataset = input_fn()
        self._get_widths(dataset)
        self._build_model()

        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        print("Dataset size: "+str(dataset.output_shapes))
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        subtotal = 0
        total_tested = 0

        with tf.Session() as sess:
            if self.model_dir:
                saver = tf.train.Saver()
                saver.restore(sess, self.model_dir)
                print("Model restored.")

            batch_num = 0

            # Build graph to calculate accuracy
            label_class = tf.argmax(self.y, 1)
            correct_preds = tf.equal(self.maxpreds, label_class)
            correct = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

            import timeit
            start = timeit.default_timer()

            while True:
                try:
                    xs, ys = sess.run(next_element)

                    total_tested += xs.shape[0]

                    subtotal += correct.eval({self.x: xs, self.y: ys})

                    if batch_num % display_period == 0:
                        print("Batch: %06d %d/%d" % (batch_num, subtotal, total_tested))

                    batch_num += 1

                except tf.errors.OutOfRangeError:
                    print("End of dataset.")
                    break

            stop = timeit.default_timer()

        print("Finished evaluation: %.5f%%" % (subtotal/total_tested))
        print(stop - start)

    def predict(self, docs, num_classes=10):
        tf.reset_default_graph()

        self.width_in = len(self.word_dict.keys())
        self.width_out = num_classes

        self._build_model()

        inputs = np.zeros([len(docs), len(self.word_dict.keys())])
        for i, doc in enumerate(docs):
            for j, word in enumerate(word_tokenize(doc)):
                if word in self.word_dict:
                    inputs[i][j] = 1

        with tf.Session() as sess:
            saver = tf.train.Saver()
            if self.model_dir:
                saver.restore(sess, self.model_dir)
                self.model_dir = None

            label_classes = sess.run(self.maxpreds, feed_dict={self.x: inputs})

        return label_classes

