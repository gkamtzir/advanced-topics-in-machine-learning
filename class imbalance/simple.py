from utilities import load_data, logistic_regression, explicit_random_forest

# Loading data
X, y = load_data()

# Creating and evaluating the model

# Logistic regression
logistic_regression(X, y)

# Random forest
explicit_random_forest(X, y)