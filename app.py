from flask import Flask,render_template,url_for,request
app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=["POST"])
def predict():
    model_name,log_file=None,None
    if request.method=='POST':
        model_name=request.form['model-name']
        log_file=request.files['file-dir']

    if model_name=="Encoder":
        pass



if __name__ == '__main__':
    app.run()
