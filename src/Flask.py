import argparse
from flask import Flask, render_template, request
from models.logreg import *
app = Flask(__name__)

parser = argparse.ArgumentParser(description='Movie Review Classifier')
# parser.add_argument('-dataset', type=str, default='dataset/dataset_full.npz', help='dataset to use')
# parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
# parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train [default: 10]')
# parser.add_argument('-batch-size', type=int, default=1, help='batch size')
# parser.add_argument('-hidden-size', type=int, default=200, help='size of hidden state')
# parser.add_argument('-max-timesteps', type=int, default=200, help='length of sequence tracked')
# parser.add_argument('-embedding-dim', type=int, default=100, help='batch size')
# parser.add_argument('-log-interval', type=int, default=1, help='logging rate')
# parser.add_argument('-vocab-size', type=int, default=87798, help='number of words to embed')
# parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# parser.add_argument('-num_classes', type=int, default=10, help='number of classes being predicted')
parser.add_argument('-model-dir', type=str, default='snapshot/logres.chpt', help='filename of model snapshot [default: None]')
# parser.add_argument('-predict', type=str, default='', help='predict the results for the dataset')
# parser.add_argument('-test', action='store_true', default=False, help='train or test')
# parser.add_argument('-display-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
# parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
# parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
# parser.add_argument('-model', type=str, default="lstm", help='model to use. e.g. lstm, linreg')
args = parser.parse_args()
log_reg = LogisticRegression(args)


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
