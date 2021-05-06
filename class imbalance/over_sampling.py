from imblearn.over_sampling import RandomOverSampler
from utilities import load_data, logistic_regression

# Loading data
X, y = load_data()

# Performing random over-sampling
ros = RandomOverSampler(random_state = 0)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Creating and evaluating the model
logistic_regression(X_resampled, y_resampled)