#!/bin/sh

if command -v md5sum 2>/dev/null; then
    MD5="md5sum" 
else 
    MD5="md5"
fi

cd data
if [ ! -f glove.6B.100d.txt ]; then
    if ! $MD5 glove.6B.zip | grep -q 056ea991adb4740ac6bf1b6d9b50408b; then
        curl -OL http://nlp.stanford.edu/data/glove.6B.zip
    fi
    echo "zip file present"
    echo "unzipping embeddings"
    unzip glove.6B.zip glove.6B.100d.txt
else
    echo "embeddings present"
fi
