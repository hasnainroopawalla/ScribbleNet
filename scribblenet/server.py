from flask import Flask, render_template, request, jsonify

from scribblenet.ml.prediction import predict
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)


@app.route("/")
def firstpage():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@cross_origin()
def doodle_predict():
    image_string = request.form["image_string"]
    return jsonify({"prediction": predict(image_string)})


if __name__ == "__main__":
    app.run(debug=False)
