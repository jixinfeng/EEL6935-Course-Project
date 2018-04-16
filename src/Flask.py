from flask import Flask, render_template, request
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        form_data = request.form
        display_data = {"Model": form_data["model"],
                        "Review": form_data["review"],
                        "Granularity": form_data["granularity"]}
        return render_template("result.html", result=display_data)


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=80)
