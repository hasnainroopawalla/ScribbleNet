from flask import Flask, render_template, request, jsonify

from scribblenet.ml.prediction import predict
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)


@app.route("/")
def index():
    """Renders the front-end for the server-based UI.
    """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@cross_origin()
def doodle_predict():
    """Predicts a class for the doodle image string.

    Returns:
        Dict[str, List[str]]: A List of predicted classes ordered by probability.
    """
    image_string = request.form["image_string"]
    prediction = predict(image_string)
    print(prediction)
    return jsonify({"prediction": prediction})


@app.route("/wakeup", methods=["GET"])
@cross_origin()
def server_wakeup():
    """Performs a basic activity on the server to wake it up (Heroku's inactive dyno workaround).
    """
    print("Waking up..")
    return ("", 204)


if __name__ == "__main__":
    app.run(debug=False)
