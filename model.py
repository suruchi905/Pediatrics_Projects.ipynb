# model.py
from river import compose, preprocessing, linear_model
from river.preprocessing import OneHotEncoder

# Define a simple online model (Logistic Regression)
model = compose.Pipeline(
    OneHotEncoder(),
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression()
)

def predict(model, x):
    return model.predict_one(x)

def learn(model, x, y):
    model.learn_one(x, y)
