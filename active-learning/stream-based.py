import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from modAL.uncertainty import classifier_uncertainty
from modAL.models import ActiveLearner

# Loading data
data = pd.read_csv('creditcard.csv')

# Initializing scaler
robust_scaler = RobustScaler()

# Scale 'Amount' and 'Time' features
data['scaled_amount'] = robust_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data['scaled_time'] = robust_scaler.fit_transform(data['Time'].values.reshape(-1,1))
data.drop(['Time','Amount'], axis=1, inplace=True)

# Moving the 'Class' column to the end of the data frame
class_column = data.pop('Class')
data = pd.concat([data, class_column], 1)
    
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

def stream_based(X, y, estimator, figure_name):
    # Undersampling the data
    under = RandomUnderSampler()
    
    X, y = under.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    X = X.values
    y= y.values
    
    # Training the initial model with only 5 random instances
    n_initial = 5
    initial_idx = np.random.choice(range(len(X)), size=n_initial, replace=False)
    X_train_initial, y_train_initial = X[initial_idx], y[initial_idx]
    
    learner = ActiveLearner(
        estimator = estimator,
        X_training = X_train_initial, y_training = y_train_initial
    )
    
    unqueried_score = learner.score(X_train, y_train)
    
    performance_history = [unqueried_score]
    accuracy_scores = []
    precision_scores = []
    f1_scores = []
    recall_scores = []
    
    failed_counter = 0
    # Continue learning until the learner reaches a score of 0.971
    while learner.score(X_train, y_train) < 0.97:
        stream_idx = np.random.choice(range(len(X_train)))
        if classifier_uncertainty(learner, X_train[stream_idx:stream_idx + 1]) >= 0.4:
            failed_counter = 0
            learner.teach(X_train[stream_idx:stream_idx + 1], y_train[stream_idx:stream_idx + 1])
            new_score = learner.score(X_train, y_train)
            performance_history.append(new_score)
            y_pred = learner.predict(X_test)
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))
            recall_scores.append(recall_score(y_test, y_pred))
            print(f'Instance no. {stream_idx} queried, new accuracy: {new_score}')
        else:
            failed_counter += 1
        if failed_counter > 10000:
            break
            
    # Plotting results
    plt.plot([i for i in range(len(accuracy_scores))], accuracy_scores, label = 'Accuracy')
    plt.plot([i for i in range(len(precision_scores))], precision_scores, label = 'Precision')
    plt.plot([i for i in range(len(f1_scores))], f1_scores, label = 'F1')
    plt.plot([i for i in range(len(recall_scores))], recall_scores, label = 'Recall')
    plt.plot([i for i in range(len(performance_history))], performance_history, label = 'Score')
    plt.xlabel('Number of queried instances')
    plt.ylabel('Scores')
    plt.legend()
    plt.savefig(f'{figure_name}.png')
    
# Random Forest
stream_based(X, y, RandomForestClassifier(), 'random_forest')

# Logistic Regression
stream_based(X, y, LogisticRegression(), 'logistic_regression')