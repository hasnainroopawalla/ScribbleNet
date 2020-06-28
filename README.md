# Doodle-Classifier
The notebook includes downloading the classes.txt file as well as the entire dataset of 'Quick Draw' images

The 100_classes.txt consists of 100 common classes (a subset of the 345 original classes)

The number of images used per class is limited to 16000 to increase training speed

With my architecture I achieved a Validation Accuracy of 96%


# Instructions to train the model:

Run the Doodle.ipynb on Google Colab (GPU for more disk memory)

# Instructions for Flask:
In the command prompt:
```
python run.py
```
Now, Navigate to 'localhost:5000' on your browser

# Test Results:

Suitcase:

![Suitcase](https://github.com/hasnainroopawalla/Doodle-Classifier/blob/master/images/suitcase.gif)

Headphones:

![Headphones](https://github.com/hasnainroopawalla/Doodle-Classifier/blob/master/images/headphones.gif)

Lightning:

![Lightning](https://github.com/hasnainroopawalla/Doodle-Classifier/blob/master/images/lightning.gif)

Fan:

![Fan](https://github.com/hasnainroopawalla/Doodle-Classifier/blob/master/images/fan.gif)

Car:

![Car](https://github.com/hasnainroopawalla/Doodle-Classifier/blob/master/images/car.gif)
