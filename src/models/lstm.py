import tensorflow as tf
import numpy as np
import datetime
import timeit
from nltk import word_tokenize
from collections import defaultdict
import pandas as pd


# Built referencing https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py
# and https://web.stanford.edu/class/cs20si/2017/lectures/notes_03.pdf

def data_type():
    return tf.float32


class LSTM:
    def __init__(self, args):
        self.args = args
        self.loss = None
        self.graph = tf.Graph()

        if hasattr(self.args, 'word_dict'):
            self.word_dict = defaultdict(lambda: 1, self.args.word_dict)
        else:
            self.word_dict = None

        self.sess = tf.Session(graph=self.graph)

        if args.model_dir:
            self.model_dir = args.model_dir
            self.loss, self.train_step = self._build_model()
            with self.graph.as_default():
                saver = tf.train.Saver()
                saver.restore(self.sess, args.model_dir)
                print("Model restored.")

    def __del__(self):
        if self.sess:
            self.sess.close()

    def _get_lstm_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(
            self.args.hidden_size, forget_bias=1.0, state_is_tuple=True,
            reuse=False)

    def _sequence_lengths(self, input_tensor):
        # input - <timesteps, batch_size> - contains word indexes.
        # index 0 = sentence end
        lengths = tf.reduce_sum(tf.sign(input_tensor), axis=1) # find number of words per batch item
        return lengths

    def _load_embeddings(self, embedding_file, sess):
        with self.graph.as_default():
            tf_weight_placeholder = tf.placeholder(data_type(), [self.args.vocab_size, self.args.embedding_dim])
            [weights] = sess.run([self.embedding_weights])
            update_op = tf.assign(self.embedding_weights, weights)

        count = 0
        print("Loading embeddings")
        print("Vocab size:", self.args.vocab_size)

        with open(embedding_file) as f:
            for line_num, line in enumerate(f.readlines()):
                if (line_num+1) % 100000 == 0:
                    print("Line:", line_num)
                items = line.split()
                word = items[0]
                if word in self.word_dict:
                    count += 1
                    ix = self.word_dict[word]
                    weights[ix] = [float(item) for item in items[1:]]

        sess.run(update_op, feed_dict={tf_weight_placeholder: weights})

    # Returns a tensorflow variable containing the loss function
    def _build_model(self):
        print("Batch size:",self.args.batch_size)
        with self.graph.as_default():
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
                print("inputs len:", len(inputs))

                self.lengths = self._sequence_lengths(self.inputs)
                outputs, state = tf.nn.static_rnn(self.lstm_cell, inputs,
                                                  sequence_length=self.lengths,
                                                  initial_state=self._initial_state)

                index = tf.range(0, self.args.batch_size) * self.args.max_timesteps + (self.lengths - 1)
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

            # Create optimization step
            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(self.args.lr)
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
                    # Embeddings weights are too large to add apparently - grad is a sparse matrix or something
                    if var.name != "embedding_weights:0":
                        param_summaries.append(tf.summary.histogram(var.name + '/gradient', grad))

        return loss, train_step

    # Train the model with the Dataset passed in
    def train(self, input_fn, args):
        print("Beginning to Train")
        start_train = timeit.default_timer()
        dataset = input_fn()

        # Batch the data
        dataset = dataset.shuffle(buffer_size=25000)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.args.batch_size))
        dataset = dataset.prefetch(1)

        # Prepare our model and acquire a reference to the loss
        if self.loss is None:
            self.loss, self.train_step = self._build_model()
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self._load_embeddings('../data/glove.6B.100d.txt', self.sess)

        # Get an iterator to the dataset
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

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


        # Init graph variables and set up logging
        train_writer = tf.summary.FileWriter('./logs/train/%s' % datetime.datetime.now(), self.sess.graph)

        # Log hyperparameters
        summaries = self.sess.run([batch_size_summary, learning_rate_summary, hidden_size_summary, embedding_dim_summary],
                             feed_dict={
                                 batch_size_node: args.batch_size,
                                 learning_rate_node: args.lr,
                                 hidden_size_node: args.hidden_size,
                                 embedding_dim_node: args.embedding_dim
                             })

        for summary in summaries:
            train_writer.add_summary(summary, 0)

        step_num = 0
        saver = tf.train.Saver()

        print("Starting to Train")

        for epoch in range(args.epochs):
            self.sess.run(iterator.initializer)
            total_cost = 0.
            batch_num = 1
            running_cost = 0.

            start = timeit.default_timer()

            # Run through the epoch
            while True:
                try:
                    # Note that each x, y pair in the batch has an epoch number
                    inputs, labels = self.sess.run(next_element)

                except tf.errors.OutOfRangeError:
                    print("End of training dataset.")
                    break
                # Train using batch data
                summary, c, _ = self.sess.run([merged, self.loss, self.train_step], feed_dict={self.inputs: inputs,
                                                                                               self.labels: labels})

                # New step
                step_num += 1

                # Accumulate loss
                running_cost += c
                total_cost += c

                # Display log message every n batches
                if batch_num % args.display_interval == 0:
                    end = timeit.default_timer()
                    print("Epoch: %04d Batch: %06d time/batch=%.4f avg_cost=%.9f" %
                          (epoch + 1, batch_num, (end-start)/(self.args.batch_size * self.args.display_interval),
                           running_cost / args.display_interval))

                    maxpreds, label_classes = self.sess.run([self.maxpreds, self.label_class],
                                                        feed_dict={self.inputs: inputs, self.labels: labels})

                    print(label_classes)
                    print(maxpreds)

                    running_cost = 0.
                    start = timeit.default_timer()

                if step_num % args.log_interval == 0:
                    train_writer.add_summary(summary, step_num)
                batch_num += 1

            # Calculate the avg loss over the batches for this epoch
            epoch_loss = total_cost / batch_num
            print("Total cost:", total_cost)
            print("Total batches:", batch_num)
            print("Total avg loss:", epoch_loss)
            summary, = self.sess.run([epoch_loss_summary], feed_dict={epoch_loss_node: epoch_loss})
            train_writer.add_summary(summary, epoch)

            if step_num % args.save_interval == 0 and args.save_dir:
                save_path = saver.save(self.sess, args.save_dir, global_step=step_num)
                print("Finished epoch and saved model to " + save_path)

        end_train = timeit.default_timer()

        save_path = saver.save(self.sess, args.save_dir, global_step=step_num)
        print("Saved final model model to " + save_path)

        print("Training Finished after %.2f minutes" % ((end_train - start_train) / 60))
        print("Time per epoch: %.2fs" % ((end_train-start_train)/args.epochs))

    def score(self, input_fn, args):
        tf.reset_default_graph()
        dataset = input_fn()
        loss, _ = self._build_model()

        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.args.batch_size))
        print("Dataset size: "+str(dataset.output_shapes))
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        subtotal = 0
        total_tested = 0

        length_score_pred = []
        conf_matrix = np.zeros([self.args.num_classes, self.args.num_classes])

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

                    cost, maxpreds, label_classes, lengths, num_correct = sess.run([loss, self.maxpreds, self.label_class, self.lengths, correct],
                                                                          feed_dict={self.inputs: xs, self.labels: ys})

                    total_cost += cost
                    subtotal += num_correct

                    length_score_pred.append((lengths[0], label_classes[0], maxpreds[0]))
                    conf_matrix[maxpreds[0], label_classes[0]] += 1

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

        print("Finished evaluation: %.5f%%" % (100 * subtotal/total_tested))
        print("Total cost:", total_cost)
        print(stop - start)

        np.save("con_matrix", conf_matrix)
        df = pd.DataFrame(length_score_pred, columns=['len', 'score', 'pred'])
        df.to_csv('len_score_pred.csv')

        return 100 * subtotal/total_tested

    def predict(self, docs):
        assert self.word_dict is not None, "word_dict must not be None"
        inputs = np.zeros([len(docs), self.args.max_timesteps])
        for i, doc in enumerate(docs):
            for j, word in enumerate(word_tokenize(doc.replace('.',' . '). replace('-', ' - '))):
                if j == self.args.max_timesteps:
                    break
                inputs[i][j] = self.word_dict[word.lower()]

        print(inputs)
        label_classes = self.sess.run(self.maxpreds, feed_dict={self.inputs: inputs})

        return label_classes

