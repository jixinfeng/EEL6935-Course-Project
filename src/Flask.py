import argparse
import numpy as np
from types import SimpleNamespace
from flask import Flask, render_template, request
from models.logreg import *
from models.lstm import *
app = Flask(__name__)

config = LRConfig(width_out=10, vocab_file='data/logreg/imdb.vocab')
log_reg = LogisticRegression(config,  model_dir='trained/logreg')

word_dict = np.load('data/lstm/train_dict.npy').item()
args = SimpleNamespace(
            batch_size=1,
            max_timesteps=200,
            model_dir='trained/lstm',
            log_interval=1000,
            num_classes=10,
            vocab_size=43481,
            embedding_dim=100,
            hidden_size=200,
            word_dict=word_dict,
            display_interval=500,
            lr=0.001
        )
lstm = LSTM(args)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        form_data = request.form
        review = form_data["review"]	

        if form_data["model"] == "baseline":
            score = log_reg.predict([review])[0]
        else:
            score = lstm.predict([review])[0]

        display_data = {"Model": form_data["model"],
                        "Granularity": form_data["granularity"],
                        "Sentiment": score}
        return render_template("result.html", result=display_data)


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=80)
