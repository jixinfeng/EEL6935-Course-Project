import tensorflow as tf
import numpy as np
import datetime
import timeit
from nltk import word_tokenize
from collections import defaultdict

# Built referencing https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py
# and https://web.stanford.edu/class/cs20si/2017/lectures/notes_03.pdf

def data_type():
    return tf.float32


class LSTM:
    def __init__(self, args):
        self.args = args

        if hasattr(self.args, 'word_dict'):
            self.word_dict = defaultdict(int, self.args.word_dict)
        else:
            self.word_dict = defaultdict(int)

    def _get_lstm_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(
            self.args.hidden_size, forget_bias=1.0, state_is_tuple=True,
            reuse=False)

    def _sequence_lengths(self, input_tensor):
        # input - <timesteps, batch_size> - contains word indexes.
        # index 0 = sentence end
        lengths = tf.reduce_sum(tf.sign(input_tensor), axis=1) # find number of words per batch item
        return lengths

    # Returns a tensorflow variable containing the loss function
    def _build_model(self):
        print("Batch size:",self.args.batch_size)
        with tf.name_scope('input'):
            # Input data
            self.inputs = tf.placeholder(tf.int32, shape=[self.args.batch_size, self.args.max_timesteps])
            self.labels = tf.placeholder(tf.float32, shape=[self.args.batch_size, self.args.num_classes])
            self.label_class = tf.argmax(self.labels, 1)

            # Create embedding tensor
            self.embedding_weights = tf.get_variable('embedding_weights', [self.args.vocab_size, self.args.embedding_dim], data_type())
            # Look up embeddings
            self.embed = tf.nn.embedding_lookup(self.embedding_weights, self.inputs)

        with tf.name_scope('lstm'):
            self.lstm_cell = self._get_lstm_cell()
            self._initial_state = self.lstm_cell.zero_state(self.args.batch_size, data_type())

            # get a list of timesteps
            inputs = tf.unstack(self.embed, num=self.args.max_timesteps, axis=1)
            print("inputs shape:", self.embed.shape)
            print("single timestep shape: ", inputs[0].shape)
            print("inputs len:",len(inputs))

            lengths = self._sequence_lengths(self.inputs)
            outputs, state = tf.nn.static_rnn(self.lstm_cell, inputs,
                                              sequence_length=lengths,
                                              initial_state=self._initial_state)

            index = tf.range(0, self.args.batch_size) * self.args.max_timesteps + (lengths - 1)
            flat = tf.reshape(outputs, [-1, self.args.hidden_size])
            self.outputs = tf.gather(flat, index)

        with tf.name_scope('softmax'):
            # Set model weights
            self.W = tf.get_variable("W", shape=(self.args.hidden_size, self.args.num_classes))

            self.b = tf.get_variable("b", shape=self.args.num_classes)

            self.logits = tf.matmul(self.outputs, self.W) + self.b

            #eps = tf.get_variable("eps", shape=[1, self.args.num_classes],
            #                      initializer=tf.constant_initializer(0.0000001 * np.ones([1, self.args.num_classes])),
            #                      trainable=False)

            self.preds = tf.nn.softmax(self.logits)

            self.maxpreds = tf.argmax(self.preds, 1)

        with tf.name_scope('loss') as self.loss_scope:
            # Minimize error using cross entropy
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
            tf.summary.scalar('batch_loss', loss)
            return loss

    # Train the model with the Dataset passed in
    def train(self, input_fn, args):
        print("Beginning to Train")
        start_train = timeit.default_timer()
        dataset = input_fn()

        # Prepare our model and acquire a reference to the loss
        loss = self._build_model()

        # Batch the data
        dataset = dataset.shuffle(buffer_size=25000)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.args.batch_size))

        # Get an iterator to the dataset
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        # Create optimization step
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(args.lr)
            grads = optimizer.compute_gradients(loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
            train_step = optimizer.apply_gradients(capped_gvs)

        # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_advanced.py
        with tf.name_scope('parameters'):
            param_summaries = []
            for var in tf.trainable_variables():
                param_summaries.append(tf.summary.histogram(var.name, var))
            grads = list(zip(grads, tf.trainable_variables()))
            for grad, var in grads:
                if (var.name != "embedding_weights:0"):
                    param_summaries.append(tf.summary.histogram(var.name + '/gradient', grad))
        merged = tf.summary.merge_all()

        with tf.name_scope(self.loss_scope):
            epoch_loss_node = tf.placeholder(tf.float32, shape=())
            epoch_loss_summary = tf.summary.scalar('epoch_loss', epoch_loss_node)

        with tf.name_scope('hyperparameters'):
            batch_size_node = tf.placeholder(tf.float32, shape=())
            batch_size_summary = tf.summary.scalar('batch_size', batch_size_node)

            learning_rate_node = tf.placeholder(tf.float32, shape=())
            learning_rate_summary = tf.summary.scalar('learning_rate', learning_rate_node)

            hidden_size_node = tf.placeholder(tf.float32, shape=())
            hidden_size_summary = tf.summary.scalar('hidden_unit_size', hidden_size_node)

            embedding_dim_node = tf.placeholder(tf.float32, shape=())     
            embedding_dim_summary = tf.summary.scalar('embedding_dim_size', embedding_dim_node)

        # Start training
        with tf.Session() as sess:
            # Load the previous model if it was listed in the constructor
            saver = tf.train.Saver()
            if args.model_dir:
                saver.restore(sess, args.model_dir)
            else:
                # Otherwise, initialize the model
                init = tf.global_variables_initializer()
                sess.run(init)

            # Init graph variables and set up logging
            train_writer = tf.summary.FileWriter('./logs/train/%s' % datetime.datetime.now(), sess.graph)

            # Log hyperparameters
            summaries = sess.run([batch_size_summary, learning_rate_summary, hidden_size_summary, embedding_dim_summary],
                                 feed_dict={
                                     batch_size_node: args.batch_size,
                                     learning_rate_node: args.lr,
                                     hidden_size_node: args.hidden_size,
                                     embedding_dim_node: args.embedding_dim
                                 })

            for summary in summaries:
                train_writer.add_summary(summary, 0)

            step_num = 0
            for epoch in range(args.epochs):
                sess.run(iterator.initializer)
                total_cost = 0.
                batch_num = 1
                running_cost = 0.

                start = timeit.default_timer()

                # Run through the epoch
                while True:
                    try:
                        # Note that each x, y pair in the batch has an epoch number
                        inputs, labels = sess.run(next_element)

                    except tf.errors.OutOfRangeError:
                        print("End of training dataset.")
                        break

                    # Train using batch data
                    summary, c, _ = sess.run([merged, loss, train_step], feed_dict={self.inputs: inputs,
                                                                                    self.labels: labels})

                    # Accumulate loss
                    running_cost += c
                    total_cost += c

                    # Display log message every n batches
                    if batch_num % args.display_interval == 0:
                        end = timeit.default_timer()
                        print("Epoch: %04d Batch: %06d time/batch=%.4f avg_cost=%.9f" %
                              (epoch + 1, batch_num, (end-start)/self.args.batch_size,
                               running_cost / args.display_interval))

                        preds, maxpreds, label_classes, logits = sess.run([self.preds, self.maxpreds, self.label_class, self.logits],
                                                                  feed_dict={self.inputs: inputs, self.labels: labels})

                        print(label_classes)
                        print(preds)
                        print(maxpreds)
                        print("logits: ",logits)

                        running_cost = 0.
                        start = timeit.default_timer()

                    if step_num % args.log_interval == 0:
                        train_writer.add_summary(summary, step_num)
                    batch_num += 1
                    step_num += 1

                # Calculate the avg loss over the batches for this epoch
                epoch_loss = total_cost / batch_num
                print("Total cost:", total_cost)
                print("Total batches:", batch_num)
                print("Total avg loss:", epoch_loss)
                summary, = sess.run([epoch_loss_summary], feed_dict={epoch_loss_node: epoch_loss})
                train_writer.add_summary(summary, epoch)

                if step_num % args.save_interval == 0 and args.save_dir:
                    save_path = saver.save(sess, args.save_dir, global_step=step_num)
                    print("Saved model to " + save_path)

            end_train = timeit.default_timer()

            save_path = saver.save(sess, args.save_dir, global_step=step_num)
            print("Saved final model model to " + save_path)

            print("Training Finished after %.2f minutes" % ((end_train - start_train) / 60))
            print("Time per epoch: %.2fs" % ((end_train-start_train)/args.epochs))

    def score(self, input_fn, args):
        tf.reset_default_graph()
        dataset = input_fn()
        loss = self._build_model()

        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.args.batch_size))
        print("Dataset size: "+str(dataset.output_shapes))
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        subtotal = 0
        total_tested = 0

        with tf.Session() as sess:
            if args.model_dir:
                saver = tf.train.Saver()
                saver.restore(sess, args.model_dir)
                print("Model restored.")

            print("Embedding weights:", self.embedding_weights.name)
            print(self.embedding_weights.eval())

            batch_num = 0
            correct_preds = tf.equal(self.maxpreds, self.label_class)

            # Calculate accuracy
            start = timeit.default_timer()
            correct = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            disp_start = timeit.default_timer()
            total_cost = 0
            while True:
                try:

                    xs, ys = sess.run(next_element)

                    total_tested += xs.shape[0]

                    cost, maxpreds, label_classes, num_correct = sess.run([loss, self.maxpreds, self.label_class, correct],
                                                                          feed_dict={self.inputs: xs, self.labels: ys})

                    total_cost += cost
                    subtotal += num_correct

                    if batch_num % args.display_interval == 0:
                        print(label_classes)
                        print(maxpreds)

                        disp_end = timeit.default_timer()
                        print("Batch: %06d %d/%d" % (batch_num, subtotal, total_tested))
                        print("Time/batch: %.3f" % ((disp_end-disp_start)/self.args.batch_size))
                        disp_start = timeit.default_timer()

                    batch_num += 1

                except tf.errors.OutOfRangeError:
                    print("End of dataset.")
                    break

            stop = timeit.default_timer()

        print("Finished evaluation: %.5f%%" % (subtotal/total_tested))
        print("Total cost:", total_cost)
        print(stop - start)

    def predict(self, docs):
        inputs = np.zeros(len(docs), self.args.max_timesteps)
        for i, doc in enumerate(docs):
            for j, word in enumerate(word_tokenize(doc)):
                inputs[i][j] = self.word_dict[word]

        with tf.Session() as sess:
            label_classes = sess.run(self.label_class, feed_dict={self.inputs: inputs})

        return label_classes

