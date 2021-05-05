from imblearn.under_sampling import RandomUnderSampler, NearMiss
from utilities import load_data, logistic_regression

# Loading data
X, y = load_data()

# Performing random under-sampling
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Creating and evaluating the model
logistic_regression(X_resampled, y_resampled)

# Performing NearMiss-1
near_miss_1 = NearMiss(version = 1)
X_resampled, y_resampled = near_miss_1.fit_resample(X, y)

# Creating and evaluating the model
logistic_regression(X_resampled, y_resampled)

# Performing NearMiss-2
near_miss_2 = NearMiss(version = 2)
X_resampled, y_resampled = near_miss_2.fit_resample(X, y)

# Creating and evaluating the model
logistic_regression(X_resampled, y_resampled)

# Performing NearMiss-3
near_miss_3 = NearMiss(version = 3)
X_resampled, y_resampled = near_miss_3.fit_resample(X, y)

# Creating and evaluating the model
logistic_regression(X_resampled, y_resampled)