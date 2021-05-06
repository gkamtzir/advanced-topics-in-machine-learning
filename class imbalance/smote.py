from imblearn.over_sampling import SMOTE
from utilities import load_data, logistic_regression

# Loading data
X, y = load_data()

# Performing smote sampling
smote = SMOTE(random_state = 0)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Creating and evaluating the model
logistic_regression(X_resampled, y_resampled)