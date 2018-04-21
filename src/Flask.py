import argparse
from flask import Flask, render_template, request
from models.logreg import *
app = Flask(__name__)


config = LRConfig(width_out=10, vocab_file='data/logreg/imdb.vocab')
log_reg = LogisticRegression(config,  model_dir='snapshot/logres.chpt')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        form_data = request.form
        score = log_reg.predict([form_data["review"]])[0]
        display_data = {"Model": form_data["model"],
                        "Granularity": form_data["granularity"],
                        "Sentiment": score}
        return render_template("result.html", result=display_data)


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=80)
