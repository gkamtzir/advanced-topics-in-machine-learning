from utilities import load_data, logistic_regression

# Loading data
X, y = load_data()

# Creating and evaluating the model
logistic_regression(X, y)