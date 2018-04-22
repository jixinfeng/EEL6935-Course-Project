from types import SimpleNamespace
from dataset import numpy_dataset
from models.lstm import LSTM
import tensorflow as tf
import pandas as pd

def main():
    to_check = ["checkpoints/model_embed-150000",
                "checkpoints/model_embed-175000",
                "checkpoints/model_embed-200000",
                "checkpoints/model_embed-225000",
                "checkpoints/model_embed-250000",
                "trained/lstm_20.ckpt-150000",
                "trained/lstm_20.ckpt-175000",
                "trained/lstm_20.ckpt-200000",
                "trained/lstm_20.ckpt-225000",
                "trained/lstm_20.ckpt-250000"]

    epochs = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20]

    dataset = numpy_dataset("data/lstm/valid.npz")

    X, Y = [], []
    for model_dir, epoch in zip(to_check, epochs):
        print("Epoch:", epoch)

        args = SimpleNamespace(
            batch_size=1,
            max_timesteps=200,
            model_dir=model_dir,
            log_interval=1000,
            num_classes=10,
            vocab_size=87798,
            embedding_dim=100,
            hidden_size=200,
            display_interval=500,
            lr=0.001
        )

        tf.reset_default_graph()
        model = LSTM(args)

        X.append(epoch)
        Y.append(model.score(dataset.input_fn, args))

    import matplotlib.pyplot as plt
    plt.plot(X, Y)
    plt.show()

    df = {'epoch': X, 'valid_acc': Y}
    df = pd.DataFrame(df)
    df.to_csv('train_results.csv')


if __name__ == "__main__":
    main()
