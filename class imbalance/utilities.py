import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
    precision_score, f1_score, recall_score, confusion_matrix, roc_auc_score
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss

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

def logistic_regression(X, y, sampling = None):
    X_train, X_test, y_train, y_test = sample_data(X, y, sampling)

    # Performing grid search
    log_reg_params = {"penalty": ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    
    grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params, scoring = 'balanced_accuracy')
    grid_log_reg.fit(X_train, y_train)
    
    log_reg = grid_log_reg.best_estimator_
    
    print(grid_log_reg.best_params_)
    
    print(cross_val_score(log_reg, X_train, y_train, cv = 5))
    
    y_pred = log_reg.predict(X_test)
    
    # Outputing the results
    calculate_results(y_test, y_pred)
    
def random_forest(X, y, sampling = None):
    X_train, X_test, y_train, y_test = sample_data(X, y, sampling)
    
    # Performing grid search
    random_forest_params = {"n_estimators": [10, 50, 100, 200, 300], "criterion": ["entropy", "gini"]}
    
    grid = GridSearchCV(RandomForestClassifier(), random_forest_params, scoring = "balanced_accuracy")
    grid.fit(X_train, y_train)
    
    random_forest = grid.best_estimator_
    
    print(grid.best_params_)
    
    print(cross_val_score(random_forest, X_train, y_train, cv = 5))
    
    y_pred = random_forest.predict(X_test)
    
    calculate_results(y_test, y_pred)
    
def explicit_random_forest(X, y, sampling = None):
    X_train, X_test, y_train, y_test = sample_data(X, y, sampling)
    
    random_forest = RandomForestClassifier(criterion = "entropy", n_estimators = 10)
    random_forest.fit(X_train, y_train)
    
    print(cross_val_score(random_forest, X_train, y_train, cv = 5))
    
    y_pred = random_forest.predict(X_test)
    
    calculate_results(y_test, y_pred)
    
def hybrid(X_train, y_train):    
    under = RandomUnderSampler(sampling_strategy = 0.015)

    X_train, y_train = under.fit_resample(X_train, y_train)
    
    over = SMOTE(sampling_strategy = 0.028)
    
    X_train, y_train = over.fit_resample(X_train, y_train)
    
    return X_train, y_train
    
def sample_data(X, y, sampling = None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    if sampling == "over":
        ros = RandomOverSampler(random_state = 0)
        X_train, y_train = ros.fit_resample(X_train, y_train)
    elif sampling == "under":
        rus = RandomUnderSampler(random_state = 0)
        X_train, y_train = rus.fit_resample(X_train, y_train)
    elif sampling == "near1":
        near_miss_1 = NearMiss(version = 1)
        X_train, y_train = near_miss_1.fit_resample(X_train, y_train)
    elif sampling == "near2":
        near_miss_2 = NearMiss(version = 2)
        X_train, y_train = near_miss_2.fit_resample(X_train, y_train)
    elif sampling == "near3":
        near_miss_3 = NearMiss(version = 3)
        X_train, y_train = near_miss_3.fit_resample(X_train, y_train)
    elif sampling == "smote":
        smote = SMOTE(k_neighbors = 10, n_jobs = -1, random_state = 0)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    elif sampling == "adasyn":
        adasyn = ADASYN(n_neighbors = 10, n_jobs = -1, random_state = 0)
        X_train, y_train = adasyn.fit_resample(X_train, y_train)
    elif sampling == "hybrid":
        X_train, y_train = hybrid(X_train, y_train)
        
    return X_train, X_test, y_train, y_test
    
def calculate_results(y_test, y_pred):
    # Outputing the results
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print(f"Accuracy: {accuracy}")
    print(f"Balanced Accuracy: {balanced_accuracy}")
    print(f"Precision: {precision}")
    print(f"F1: {f1}")
    print(f"Recall: {recall}")
    print(f"AUC: {auc}")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TN: {tn}")