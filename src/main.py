import argparse
import numpy as np
from dataset import *
from models.lstm import LSTM
from models.logreg import LogisticRegression, LRConfig


def get_model(args):
    if args.model == 'lstm':
        args.word_dict = np.load(args.dataset[:-4]+"_dict.npy").item()
        args.num_classes = 10
        model = LSTM(args)
    else:
        vocab_file = 'data/logreg/imdb.vocab'
        model = LogisticRegression(LRConfig(width_out=10, vocab_file=vocab_file), model_dir=args.model_dir)

    return model


def get_dataset(args):
    if args.model == 'lstm':
        dataset = numpy_dataset(args.dataset)
    else:
        dataset = bow_dataset(args.dataset)
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Movie Review Classifier')
    parser.add_argument('model', type=str, help='model to use. e.g. lstm, lonreg')
    parser.add_argument('dataset', type=str, help='dataset to use')

    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train [default: 10]')
    parser.add_argument('-batch-size', type=int, default=1, help='batch size')
    parser.add_argument('-hidden-size', type=int, default=200, help='size of hidden state')
    parser.add_argument('-max-timesteps', type=int, default=200, help='length of sequence tracked')
    parser.add_argument('-embedding-dim', type=int, default=100, help='batch size')
    parser.add_argument('-vocab-size', type=int, default=87798, help='number of words to embed')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-model-dir', type=str, default=None, help='filename of model  [default: None]')
    parser.add_argument('-predict', type=str    , default='', help='predict the results for the dataset')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    parser.add_argument('-log-interval', type=int, default=1, help='logging rate for tensorboard')
    parser.add_argument('-display-interval', type=int, default=1, help='logging for terminal messages')
    parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')

    args = parser.parse_args()
    model = get_model(args)

    if args.predict is not '':
        print(args.predict, ":", model.predict([args.predict]))
    else:
        dataset = get_dataset(args)
        preview_dataset(dataset.input_fn, num_items=1)

        print("Size of dataset:", dataset.size)

        if args.test:
            model.score(dataset.input_fn, args)
        else:
            model.train(dataset.input_fn, args)


if __name__ == "__main__":
    main()
