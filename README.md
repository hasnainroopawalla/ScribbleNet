# Doodle-Classifier
The notebook includes downloading the classes.txt file as well as the entire dataset

The 100_classes.txt consists of 100 common classes (a subset of the 345 original classes)

The number of images used per class is limited to 16000 to increase training speed

With the given architecture I was able to obtain 94.29% accuracy (Can be improved by using more training images)


# Instructions to train the model:

Run the Doodle.ipynb on Google Colab (GPU for more disk memory)


# Instructions for Flask:

In the command prompt, type 'python run.py'. Navigate to 'localhost:5000' on your browser and begin doodling!
