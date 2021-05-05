from imblearn.under_sampling import RandomUnderSampler
from utilities import load_data, logistic_regression

# Loading data
X, y = load_data()

# Performing random under-sampling
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Creating and evaluating the model
logistic_regression(X_resampled, y_resampled)