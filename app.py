from flask import Flask, render_template, request, flash
from model import Model

app = Flask(__name__)

model = Model()

@app.route("/")
def index():
	return render_template("index.html", context= 'rien')

@app.route("/submit", methods=['POST', 'GET'])
def greeter():
	context = model.predict(request.form['title'], request.form['body'])  # type: ignore
	return render_template("index.html", context = context)

if __name__ == "__main__":
    app.run()