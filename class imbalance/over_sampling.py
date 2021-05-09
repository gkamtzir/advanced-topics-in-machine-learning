from imblearn.over_sampling import RandomOverSampler
from utilities import load_data, logistic_regression, random_forest, \
    explicit_random_forest

# Loading data
X, y = load_data()

# Creating and evaluating the model
logistic_regression(X, y, "over")
explicit_random_forest(X, y, "over")