from imblearn.over_sampling import ADASYN
from utilities import load_data, logistic_regression

# Loading data
X, y = load_data()

# Performing ADASYN sampling
adasyn = ADASYN(random_state = 0)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

# Creating and evaluating the model
logistic_regression(X_resampled, y_resampled)