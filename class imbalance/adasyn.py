from imblearn.over_sampling import ADASYN
from utilities import load_data, logistic_regression

# Loading data
X, y = load_data()

for neighbors in range(5, 50, 5):
    print(f"Neighbors: {neighbors}")
    # Performing ADASYN sampling
    adasyn = ADASYN(n_neighbors = neighbors, n_jobs = -1, random_state = 0)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    
    # Creating and evaluating the model
    logistic_regression(X_resampled, y_resampled)