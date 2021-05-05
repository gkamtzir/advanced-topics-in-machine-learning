import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
    precision_score, f1_score, recall_score, confusion_matrix

def load_data():
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
    
    return X, y

def logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Performing grid search
    log_reg_params = {"penalty": ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    
    grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params, scoring = 'balanced_accuracy')
    grid_log_reg.fit(X_train, y_train)
    
    log_reg = grid_log_reg.best_estimator_
    
    y_pred = log_reg.predict(X_test)
    
    # Outputing the results
    calculate_results(y_test, y_pred)
    
def calculate_results(y_test, y_pred):
    # Outputing the results
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print(f"Accuracy: {accuracy}")
    print(f"Balanced Accuracy: {balanced_accuracy}")
    print(f"Precision: {precision}")
    print(f"F1: {f1}")
    print(f"Recall: {recall}")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TN: {tn}")