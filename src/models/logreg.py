import tensorflow as tf
import datetime
import numpy as np
from collections import defaultdict
from nltk import word_tokenize


# Built referencing https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py
# and https://web.stanford.edu/class/cs20si/2017/lectures/notes_03.pdf

class LRConfig():
    def __init__(self, width_out, width_in=None, vocab_file=None):
        assert width_in or vocab_file, "Either width_in or vocab_file must be given"
        self.width_out = width_out
        self.word_dict = None

        if vocab_file:
            # Load the vocab file if given
            self.word_dict = defaultdict(int)
            with open(vocab_file, encoding='utf-8') as f:
                for linenum, line in enumerate(f.readlines()):
                    self.word_dict[line[:-1]] = linenum+1
            # Safety check
            if width_in:
                assert width_in == linenum+1, "Error: width_in does not match size of vocab_file"
            self.width_in = linenum+1
        else:
            self.width_in = width_in


class LogisticRegression:
    def __init__(self, config, model_dir=None, sess=None):
        self.config = config
        self.model_dir = model_dir
        self.word_dict = self.config.word_dict

        # Create a tf session if needed
        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess

        # Load the model if one is passed in
        if self.model_dir:
            self.loss = self._build_model(config)
            saver = tf.train.Saver()
            saver.restore(self.sess, self.model_dir)
            print("Model restored.")
        else:
            self.loss = None

    def __del__(self):
        self.close_sess()

    # Return a tensorflow variable containing the loss function
    def _build_model(self, config):
        assert hasattr(config, 'width_in') and hasattr(config, 'width_out'), "Missing width_in or width_out from config"

        with tf.name_scope('input'):
            # Input data
            self.x = tf.placeholder(tf.float32, shape=[None, config.width_in])
            self.y = tf.placeholder(tf.float32, shape=[None, config.width_out])

        with tf.name_scope('weights'):
            # Set model weights
            self.W = tf.get_variable("W", shape=(config.width_in, config.width_out))

        with tf.name_scope('biases'):
            self.b = tf.get_variable("b", shape=config.width_out)

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

    # Train the model with the tf.Dataset returned by @param input_fn
    def train(self, input_fn, args):
        # Create a tf session if needed
        if self.sess is None:
            self.sess = tf.Session()

        dataset = input_fn()

        if self.loss is None:
            # Prepare our model and acquire a reference to the loss
            self.loss = self._build_model(self.config)

            # Initialize all of the model variables (i.e. assign their default values)
            init = tf.global_variables_initializer()
            self.sess.run(init)

        # Batch the data
        dataset = dataset.shuffle(buffer_size=25000)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(args.batch_size))

        # Get an iterator to the dataset
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        # Create an optimization variable
        with tf.name_scope('optimizer'):
            train_step = tf.train.AdamOptimizer(args.lr).minimize(self.loss)

        # Init graph variables and set up logging
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./logs/train/%s' % datetime.datetime.now(), self.sess.graph)

        saver = tf.train.Saver()
        step_num = 0

        for epoch in range(args.epochs):
            self.sess.run(iterator.initializer)
            batch_num = 1
            running_cost = 0.0

            # Run through the epoch
            while True:
                try:
                    # Note that each x, y pair in the batch has an epoch number
                    batch_xs, batch_ys = self.sess.run(next_element)

                except tf.errors.OutOfRangeError:
                    print("End of training dataset.")
                    break

                # Train using batch data
                summary, c, _ = self.sess.run([merged, self.loss, train_step], feed_dict={self.x: batch_xs,
                                                                                          self.y: batch_ys})
                # Accumulate loss
                running_cost += c

                # Display log message every n batches
                if batch_num % args.display_interval == 0:
                    print("Epoch: %04d Batch: %06d cost=%.9f" % (epoch + 1, batch_num, running_cost / args.display_interval))
                    running_cost = 0.0

                if step_num % args.display_interval == 0:
                    train_writer.add_summary(summary, step_num)
                batch_num += 1
                step_num += 1

            print("Training Finished!")
            if args.save_dir:
                save_path = saver.save(self.sess, args.save_dir)
                print("Saved model to "+save_path)

    def score(self, input_fn, args):
        # Create a tf session if needed
        if self.sess is None:
            self.sess = tf.Session()

        dataset = input_fn()

        if self.loss is None:
            # Prepare our model and acquire a reference to the loss
            self.loss = self._build_model(self.config)

            # Initialize all of the model variables (i.e. assign their default values)
            init = tf.global_variables_initializer()
            self.sess.run(init)

        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(args.batch_size))
        print("Dataset size: "+str(dataset.output_shapes))
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        subtotal = 0
        total_tested = 0

        batch_num = 0

        # Build graph to calculate accuracy
        label_class = tf.argmax(self.y, 1)
        correct_preds = tf.equal(self.maxpreds, label_class)
        correct = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

        import timeit
        start = timeit.default_timer()

        while True:
            try:
                xs, ys = self.sess.run(next_element)

                total_tested += xs.shape[0]

                subtotal += correct.eval({self.x: xs, self.y: ys})

                if batch_num % args.display_interval == 0:
                    print("Batch: %06d %d/%d" % (batch_num, subtotal, total_tested))

                batch_num += 1

            except tf.errors.OutOfRangeError:
                print("End of dataset.")
                break

            stop = timeit.default_timer()

        print("Finished evaluation: %.5f%%" % (subtotal/total_tested))
        print(stop - start)

    def predict(self, docs, num_classes=10):
        self.width_in = len(self.word_dict.keys())
        self.width_out = num_classes

        inputs = np.zeros([len(docs), len(self.word_dict.keys())])
        for i, doc in enumerate(docs):
            for j, word in enumerate(word_tokenize(doc)):
                if word in self.word_dict:
                    inputs[i][j] = 1

        label_classes = self.sess.run(self.maxpreds, feed_dict={self.x: inputs})

        return label_classes

    def close_sess(self):
        if self.sess:
            self.sess.close()
            self.sess = None

    def load_model(self, model_dir, config=None):
        # Reset the current session and graph
        self.close_sess()
        tf.reset_default_graph()

        # Construct
        self.loss = self._build_model()
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_dir)
        print("Model restored.")

