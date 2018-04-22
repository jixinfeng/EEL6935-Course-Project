#!/usr/bin/env bash

# Creates preprocessed datasets for training the LSTM model.
#
# Output:
#   labeled features - data_dir/lstm/test.npz
#                    - data_dir/lstm/train.npz
#
#   vocab dictionaries - data_dir/lstm/test_dict.npy
#                      - data_dir/lstm/train_dict.npy

if [ ! -d data ]; then
    echo "Directory 'data' not found. Have you run get_data.sh?"
    exit 1
fi

mkdir -p "data/lstm/"
mkdir -p "data/logreg"

echo "Generating LSTM training dataset..."
python3 lstm_preprocess.py train \
    -o data/lstm/train

echo "Generating LSTM validation dataset..."
python3 lstm_preprocess.py test \
    -num-reviews 2500 \
    -o data/lstm/valid \
    -dict data/lstm/train_dict.npy

echo "Generating LSTM testing dataset..."
python3 lstm_preprocess.py test \
    -o data/lstm/test \
    -dict data/lstm/train_dict.npy

echo "Copying files for Logistic Regression model..."
cp data/aclImdb/imdb.vocab data/logreg/imdb.vocab
cp data/aclImdb/train/labeledBow.feat data/logreg/trainBow.feat
cp data/aclImdb/test/labeledBow.feat data/logreg/testBow.feat

echo "Finished."
