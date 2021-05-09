from utilities import load_data, logistic_regression, explicit_random_forest

# Loading data
X, y = load_data()
    
# Creating and evaluating the model
logistic_regression(X, y, "adasyn")
explicit_random_forest(X, y, "adasyn")