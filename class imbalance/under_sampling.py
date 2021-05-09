from utilities import load_data, logistic_regression, random_forest, \
    explicit_random_forest

# Loading data
X, y = load_data()

# Performing random under-sampling
logistic_regression(X, y, sampling = "under")
explicit_random_forest(X, y, sampling = "under")

# Performing NearMiss-1
logistic_regression(X, y, sampling = "near1")
explicit_random_forest(X, y, sampling = "near1")

# Performing NearMiss-2
logistic_regression(X, y, sampling = "near2")
explicit_random_forest(X, y, sampling = "near2")

# Performing NearMiss-3
logistic_regression(X, y, sampling = "near3")
explicit_random_forest(X, y, sampling = "near3")