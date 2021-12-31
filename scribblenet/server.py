from flask import Flask, render_template, request, jsonify

from scribblenet.ml.prediction import predict

app = Flask(__name__)


@app.route("/")
def firstpage():
    return render_template("test.html")


@app.route("/predict", methods=["POST"])
def doodle_predict():
    image_string = request.form["image_string"]
    return jsonify({"prediction": predict(image_string)})


if __name__ == '__main__':
    app.run(debug=False)
