# news_classifier

A TensorFlow/Keras neural network model that predicts Fake news using
TextVectorization.

The model was trained on data from the [Kaggle news
dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv). The training data consists on 44270 news articles.

## data cleaning - processing

The clean data is split into training, validation and test datasets
and can be accessed in the dataset directory. The model is saved in
the `model/` directory.

## training

The model can be trained by running `train_model.ipynb`.

## testing

The model is loaded and tested by running `test_model.ipynb`. The
model can be used by running the FastAPI file `main.py`.
