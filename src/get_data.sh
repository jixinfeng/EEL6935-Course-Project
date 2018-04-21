#!/bin/sh
# Download Stanford Large Movie Review Dataset
mkdir data
cd data
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xvf aclImdb_v1.tar.gz
rm aclImdb_v1.tar.gz
echo "Data Set Download Completed."
cd ..
sh make_datasets.sh