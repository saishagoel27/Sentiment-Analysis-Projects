Project 3 Title - Sentiment Analysis with LSTM on IMDb Reviews

Project Overview

This project demonstrates how to build a sentiment analysis system using an LSTM neural network on the IMDb movie reviews dataset. The system is capable of classifying movie reviews as positive or negative.

Dataset -

The IMDb movie reviews dataset, obtained from Kaggle, is used for training and testing the model. It consists of 50,000 movie reviews, each labeled as positive or negative.

Model Architecture

The sentiment analysis model is based on an LSTM neural network. The architecture consists of the following layers:

Embedding Layer: Converts words into numerical representations.
LSTM Layer: Captures long-term dependencies in the text.
Dense Layer: Performs classification, outputting a probability for each class (positive or negative).
Implementation

The project is implemented using Python and TensorFlow/Keras. The key steps involved are:

Data Preprocessing:
Loading the dataset from the CSV file.
Tokenizing the text data.
Padding sequences to ensure uniform input length.
Model Building:
Defining the LSTM model architecture.
Compiling the model with an appropriate loss function (binary cross-entropy) and optimizer (Adam).
Model Training:
Training the model on the training data.
Evaluating the model's performance on the validation set.
Model Evaluation:
Evaluating the model's performance on the test set.
Calculating accuracy and loss metrics.
Prediction:
Creating a function to predict the sentiment of new reviews.
Tokenizing the new review, padding the sequence, and feeding it to the model.
Outputting the predicted sentiment (positive or negative).

