import numpy as np
import tensorflow as tf

# Built referencing https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py
class LogisticRegression:
    def __init__(self, model_dir=None):
        self.width_in = None
        self.width_out = None
        self.model_dir = model_dir
        self.sess = None

    def _get_widths(self, iterator):
        # Find out the input and output data dimensions
        # (Note: This information could also be passed to the constructor.
        # I'm not sure how long it takes for tensorflow to retrieve these
        # values.)
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
            W = tf.Variable(tf.zeros([self.width_in, self.width_out]))

        with tf.name_scope('biases'):
            b = tf.Variable(tf.zeros([self.width_out]))

        with tf.name_scope('softmax'):
            # Calculate prediction
            pred = tf.nn.softmax(tf.matmul(self.x, W) + b)
            self.maxpred = tf.argmax(pred, 1)

        with tf.name_scope('loss'):
            # Minimize error using cross entropy
            loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(pred), reduction_indices=1))
            tf.summary.scalar('loss_summary', loss)
            return loss

    # Train the model with the Dataset passed in
    def train(self, input_fn, num_epochs=25, learning_rate=0.0001, batch_size=20, display_period=100):
        # Prepare the dataset epochs/batches using the method suggested at
        # https://stackoverflow.com/questions/47410778/epoch-counter-with-tensorflow-dataset-api
        # to keep track of the epoch number. It's not pretty, but it'll work for now.
        dataset = input_fn()
        self._get_widths(dataset.make_one_shot_iterator())

        epoch_counter = tf.data.Dataset.range(num_epochs)
        dataset = epoch_counter.flat_map(lambda i: tf.data.Dataset.zip(
            (tf.data.Dataset.from_tensors(i).repeat(), dataset)))
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

        # Get an iterator to the dataset
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        # Prepare our model and acquire a reference to the loss
        loss = self._build_model()

        # Create an optimization variable
        with tf.name_scope('optimizer'):
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # Initialize all of the model variables (i.e. assign their default values)
        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        import datetime

        # Start training
        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('./logs/train/%s' % datetime.datetime.now(), sess.graph)
            sess.run(init)

            running_cost = 0.0
            batch_num = 0
            epoch = 0
            i = 0
            # Keep training as long as we have data
            while True:
                try:
                    # Note that each x, y pair in the batch has an epoch number
                    epochs, (batch_xs, batch_ys) = sess.run(next_element)

                    # Reset batch numbers at the beginning of a new epoch
                    if epochs[0] != epoch:
                        batch_num = 1
                    batch_num += 1
                    epoch = epochs[0]

                except tf.errors.OutOfRangeError:
                    print("End of training dataset.")
                    break

                # Train using batch data
                summary, c, _ = sess.run([merged, loss, train_step], feed_dict={self.x: batch_xs,
                                                              self.y: batch_ys})

                # Accumulate loss
                running_cost += c

                # Display log message every n batches
                if batch_num % display_period == 0:
                    print("Epoch: %04d Batch: %06d cost=%.9f" % (epoch + 1, batch_num, running_cost/display_period))
                    running_cost = 0.0

                train_writer.add_summary(summary, i)
                i += 1

            print("Training Finished!")
            save_path = saver.save(sess, "/tmp/model.ckpt")

    def binary_accuracy(self, input_fn, batch_size=20, display_period=100):
        tf.reset_default_graph()
        dataset = input_fn()
        self._get_widths(dataset.make_one_shot_iterator())
        self._build_model()

        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        print("Dataset size: " + str(dataset.output_shapes))
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        saver = tf.train.Saver()

        subtotal = 0
        total_tested = 0

        if self.sess is None:
            with tf.Session() as sess:
                saver.restore(sess, "/tmp/model.ckpt")
                print("Model restored.")

                batch_num = 0
                five = tf.constant(5, tf.int64)

                import timeit
                start = timeit.default_timer()

                while True:
                    try:
                        pred_negative = tf.less_equal(self.maxpred, five)
                        real_negative = tf.less_equal(tf.argmax(self.y, 1), five)

                        correct_prediction = tf.equal(pred_negative, real_negative)

                        # Calculate number correct
                        correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

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

                print("Finished evaluation: %.5f%%" % (subtotal / total_tested))
                print(stop - start)

    def class_accuracy(self, input_fn, batch_size=20, display_period=100):
        tf.reset_default_graph()
        dataset = input_fn()
        self._get_widths(dataset.make_one_shot_iterator())
        self._build_model()

        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        print("Dataset size: "+str(dataset.output_shapes))
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        saver = tf.train.Saver()

        subtotal = 0
        total_tested = 0

        if self.sess is None:
            with tf.Session() as sess:
                saver.restore(sess, "/tmp/model.ckpt")
                print("Model restored.")

                batch_num = 0

                import timeit
                start = timeit.default_timer()

                while True:
                    try:
                        correct_prediction = tf.equal(self.maxpred, tf.argmax(self.y, 1))

                        # Calculate accuracy
                        correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

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
