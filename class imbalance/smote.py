from imblearn.over_sampling import SMOTE
from utilities import load_data, logistic_regression

# Loading data
X, y = load_data()

for neighbors in range(5, 50, 5):
    print(f"Neighbors: {neighbors}")
    # Performing smote sampling
    smote = SMOTE(k_neighbors = neighbors, n_jobs = -1, random_state = 0)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Creating and evaluating the model
    logistic_regression(X_resampled, y_resampled)