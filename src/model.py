import numpy as np
import tensorflow as tf

class LogisticRegression:
    def __init__(self):
        self.width_in = None
        self.width_out = None

    # Returns a tensorflow variable containing the loss function
    def _build_model(self):
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, shape=[None, self.width_in])
            self.y = tf.placeholder(tf.float32, shape=[None, self.width_out])

            # Set model weights
            W = tf.Variable(tf.zeros([self.width_in, self.width_out]))
            b = tf.Variable(tf.zeros([self.width_out]))

        with tf.name_scope('softmax'):
            # Construct model
            pred = tf.nn.softmax(tf.matmul(self.x, W) + b)

        with tf.name_scope('loss'):
            # Minimize error using cross entropy
            return tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(pred), reduction_indices=1))

    def train(self, dataset, num_epochs=25, learning_rate=0.01, batch_size=20):
        # Prepare the dataset batches
        epoch_counter = tf.data.Dataset.range(num_epochs)
        dataset = epoch_counter.flat_map(lambda i: tf.data.Dataset.zip(
            (tf.data.Dataset.from_tensors(i).repeat(), dataset)))
        print "Counter Shaped", dataset.output_shapes
        batched = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

        print "Shapes: ", batched.output_shapes

        # Get an iterator to the dataset
        iterator = batched.make_initializable_iterator()
        next_element = iterator.get_next()

        # Find out the input and output data dimensions
        # (Note: This information could also be passed to the constructor.
        # I'm not sure how long it takes for tensorflow to retrieve these
        # values.)
        with tf.Session() as sess:
            sess.run(iterator.initializer)
            _, (batch_xs, batch_ys) = sess.run(next_element)
            # Network depth, # of output classes
            print("Network input depth:", batch_xs.shape[1])
            print("Network output depth:", batch_ys.shape[1])

            self.width_in = batch_xs.shape[1]
            self.width_out = batch_ys.shape[1]

        # Build the tensorflow model and acquire reference to the loss
        loss = self._build_model()

        # Create an optimization variable
        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # Initialize all of the model variables (i.e. assign their default values)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:
            sess.run(iterator.initializer)
            sess.run(init)

            running_cost = 0.0
            batch_num = 0
            epoch = 0

            # Keep training as long as we have data
            while True:
                try:
                    epochs, (batch_xs, batch_ys) = sess.run(next_element)
                    if epochs[0] != epoch:
                        batch_num = 1
                    batch_num += 1
                    epoch = epochs[0]

                except tf.errors.OutOfRangeError:
                    print("End of training dataset.")
                    break

                # Train using batch data
                _, c = sess.run([optimizer, loss], feed_dict={self.x: batch_xs,
                                                              self.y: batch_ys})
                # Compute average loss
                running_cost += c

                # Display logs every thousand batch
                if batch_num % 100 == 0:
                    print("Epoch: %04d Batch: %06d cost=%.9f" % (epoch + 1, batch_num, running_cost))
                    running_cost = 0.0

            print("Training Finished!")