from flask import Flask, render_template, url_for, request

from loganomaly.main import loganomaly_run
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

        if model_name == "Encoder":
            file = request.files.get('file-dir')
            content = file.read().decode("utf-8")
            log_list = content.split("\n")

            result_log_str,result_dict = transformer_test(log_list)

            return render_template('result.html',log_str=result_log_str, table_dict=result_dict)

        elif model_name == "LogAnomaly":
            # 娄天宇组把代码加在这里,下面是文件输入
            # 这部分需要你事先预装Flask，请自行pip install一下
            # 这部分的函数触发将在选择了日志文件，点击了predict按钮后运行
            # 文件的导入在home.html里面点击选择文件按钮，选择你的文件就可以了
            # ！！！如果你的文件格式不是.txt或.log，请在home.html第55行代码的accept处添加你的文件格式以供支持
            file = request.files.get('file-dir')
            print(file)
            #loganomaly_run(file)
            #content = file.read().decode("utf-8")
            #print(content)
            #log_list = content.split("\n")

            #要生成的两个变量：
            # 变量1. 一个保存了所有异常日志的字符串(Encoder中的result_log_str)，
            # 你可以先用列表result_list存，然后用"\n".join(result_list)生成这个字符串

            # 变量2. 一个保存了你要返回的参数的字典（比如:{"准确率": "92.8%"})
            # 要求key和value值的类型都是str

            # TODO: 你的代码

            #(变量1和变量2参考上面Encoder部分的result_log_str,result_dict）
            '''样例代码：
                result_list=[i for i in log_list]
                result_str="\n".join(result_list)

                result_dict=dict()
                result_dict["异常日志数量"]=str(int(len(result_list)*acc))
                result_dict["测试准确率"]=str(format(acc,".2%"))
            '''
            #你的最终输出参考上面Encoder部分中的return值
            #render_template('result.html',log_str=变量1, table_dict=变量2)




        elif model_name == "LogTransfer":
            #郑逸玺组把代码加在这里,下面是文件输入
            file = request.files.get('file-dir')
            content = file.read().decode("utf-8")
            log_list = content.split("\n")
            # 这部分需要你事先预装Flask，请自行pip install一下
            # 这部分的函数触发将在选择了日志文件，点击了predict按钮后运行
            # 文件的导入在home.html里面点击选择文件按钮，选择你的文件就可以了
            # ！！！如果你的文件格式不是.txt或.log，请在home.html第55行代码的accept处添加你的文件格式以供支持

            # TODO: 你的代码

            #要生成的两个变量：
            # 变量1. 一个保存了所有异常日志的字符串(Encoder中的result_log_str)，
            # 你可以先用列表result_list存，然后用"\n".join(result_list)生成这个字符串

            # 变量2. 一个保存了你要返回的参数的字典（比如:{"准确率": "92.8%"})
            # 要求key和value值的类型都是str

            #(变量1和变量2参考上面Encoder部分的result_log_str,result_dict）
            '''样例代码：
                result_list=[i for i in log_list]
                result_str="\n".join(result_list)

                result_dict=dict()
                result_dict["异常日志数量"]=str(int(len(result_list)*acc))
                result_dict["测试准确率"]=str(format(acc,".2%"))
            '''
            #你的最终输出参考上面Encoder部分中的return值
            #render_template('result.html',log_str=变量1, table_dict=变量2)


if __name__ == '__main__':
    app.run()
