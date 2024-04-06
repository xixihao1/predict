from flask import Flask, render_template

app = Flask(__name__)
app.config['SECRET_KEY'] = '1456719640@qq.com'


@app.route("/")
def root():
    """
    主页
    :return: Index.html
    """
    return render_template('Index.html')


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.2', port='5000')
