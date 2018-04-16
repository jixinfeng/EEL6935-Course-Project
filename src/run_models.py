import numpy as np
import tensorflow as tf

from lstm import LSTM


def demo_dataset(inputs, labels):
    print(inputs[0], labels[0])

def main():
    # Create a logistic regression model,
    # and train on the Stanford Movie Review Dataset
    with np.load('normal.sample.npz') as data:
        inputs = data['inputs']
        labels = data['labels']

    lrs = [0.001, 0.1, 0.0001]
    batchs = [1, 10, 50, 100]
    hiddens = [10, 100, 200, 400, 800]

    demo_dataset(inputs, labels)
    load_dataset = lambda: tf.data.Dataset.from_tensor_slices((inputs, labels))

    for lr in lrs:
        for batch in batchs:
            for hidden in hiddens:
                print(lr, batch, hidden)
                model = LSTM(batch_size=batch, hidden_size=hidden)
                model.train(load_dataset, save_dir='checkpoints/model.ckpt', num_epochs=50, learning_rate=lr,
                            display_period=100)

if __name__ == '__main__':
    main()