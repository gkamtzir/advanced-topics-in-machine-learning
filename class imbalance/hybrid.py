from utilities import load_data, logistic_regression, \
    explicit_random_forest

# Loading data
X, y = load_data()

# Creating and evaluating the model
logistic_regression(X, y, "hybrid")
explicit_random_forest(X, y, "hybrid")