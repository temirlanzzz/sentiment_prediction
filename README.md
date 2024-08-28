# sentiment_prediction
Sentiment prediction trained on IMDB movie reviews dataset with FastAPI for inference using custom Transformer Architecture from Scratch

This repository contains a sentiment analysis model implemented using a Transformer-based architecture in Keras. The model is trained to predict the sentiment of movie reviews as either positive or negative. The approach utilizes a custom tokenizer and positional encoding to handle textual data effectively.

## Table of Contents

- [Overview](#overview)
- [Approach](#approach)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training and evaluating the Model](#training-and-evaluating-the-model)
- [Using the Model in FastAPI](#using-the-model-in-fastapi)
- [Results](#results)

## Overview

This project demonstrates a sentiment analysis model using a Transformer-based architecture. The model processes input reviews to classify them into either "positive" or "negative" sentiments. The project is divided into two main components:
1. Training and evaluating the model using Jupyter Notebook.
2. Deploying the trained model in a FastAPI application to provide a prediction endpoint.

## Approach

The model uses a Transformer-based architecture with an embedding layer, positional encoding, and transformer blocks to process text data. The steps in the approach are:

1. **Tokenization and Preprocessing**: The text data is tokenized using a Keras Tokenizer and converted into sequences. Positional encoding is added to handle the order of words in the sequences.
2. **Model Architecture**: The model consists of an embedding layer, positional encoding, transformer blocks (multi-head attention and feed-forward layers), global average pooling, and a final dense layer for binary classification.
3. **Training**: The model is trained on a dataset of movie reviews using binary cross-entropy loss and the Adam optimizer.
4. **Evaluation**: The model's performance is evaluated on a validation set to ensure it generalizes well to unseen data.
5. **Deployment**: The trained model is deployed using FastAPI to serve predictions via an HTTP API.

## Requirements

- Python 3.7 or higher
- Jupyter Notebook
- TensorFlow / Keras
- FastAPI
- Uvicorn
- Pandas
- Numpy
- Scikit-learn
- Pickle

## Data Preparation

Before using dataset for sentiment analysis, it had to be cleaned and preprocessed:
- Convert to lowercase
- Remove HTML tags
- Remove non-alphabetic characters
- Remove stopwords using nltk.corpus.stopwords.words('english')
- Encode output labels 'positive': 1, 'negative': 0
- Using Keras Tokenizer and fit on text data

## Model Architecture
- Multi-Head Self-Attention Layer: allows the model to focus on different parts of the input sequence simultaneously by computing attention scores for multiple representation subspace
- Positional Encoding Layer: introduce information about the position of words in the sequence since transformers are inherently permutation invariant
- Transformer Block: captures complex relationships in the input data by stacking multiple layers of attention and feed-forward neural networks.
- Total params: 2,857,473

## Training and Evaluating the Model
To train the model:
- Load the dataset, tokenize the text, and prepare the sequences for training.
- Construct the Transformer-based model architecture using Keras.
- Compile the model with a binary cross-entropy loss function and the Adam optimizer, then train for 5 epochs.
- Evaluate the model on test set.
- Predict sentiment on custom text.

## Using the Model in FastAPI
- Load the model and tokenizer.
- Front-end page (static/index.html) and endpoint for prediction (main.py).
- Separate classes for loading model in layers.py.
- Run the FastAPI server using Uvicorn: uvicorn main:app --reload
- Open app on http://127.0.0.1:8000
- Enter review text and press "Predict Sentiment" button.
- Result will be shown below.

## Results 
Model accuracy trained on 5 epochs:
![alt text](https://github.com/temirlanzzz/sentiment_prediction/blob/main/accuracy.png?raw=true)
Model loss trained on 5 epochs:
![alt text](https://github.com/temirlanzzz/sentiment_prediction/blob/main/loss.png?raw=true)
