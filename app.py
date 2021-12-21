from flask import Flask, render_template, url_for, request
from transformer.main import transformer_test

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict', methods=["POST"])
def predict():


    model_name, log_file = None, None
    if request.method == 'POST':
        model_name = request.form['model-name']
        # log_file = request.files['file-dir']

        if model_name == "Encoder":
            # if model_name == "Encoder":
            file = request.files.get('file-dir')
            content = file.read().decode("utf-8")
            log_list = content.split("\n")

            result_log_str,result_dict = transformer_test(log_list)

            return render_template('result.html',log_str=result_log_str, table_dict=result_dict)



if __name__ == '__main__':
    app.run()
