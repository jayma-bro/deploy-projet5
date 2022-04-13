from flask import Flask, render_template, request, flash
from model import Model

app = Flask(__name__)

model = Model()

@app.route("/")
def index():
	return render_template("index.html", context='rien')

@app.route("/submit", methods=['POST', 'GET'])
def greeter():
	return render_template("index.html", context=model.predict(request.form['title'], request.form['body']))  # type: ignore

if __name__ == "__main__":
    app.run(debug=True)