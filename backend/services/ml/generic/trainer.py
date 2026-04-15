from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats import randint

def train_model(X, y, problem_type: str, perform_tuning: bool = False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_params = "Default"
    
    if problem_type == "Klasifikasi":
        model = RandomForestClassifier(random_state=42)
        model_name = "RandomForestClassifier"

        if perform_tuning:
            param_dist = {
                'n_estimators': randint(100, 500),
                'max_depth': [10, 20, 30, None],
                'min_samples_leaf': randint(1, 4)
            }
            search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42, n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = search.best_params_

    elif problem_type == "Regresi":
        model = RandomForestRegressor(random_state=42)
        model_name = "RandomForestRegressor"

        if perform_tuning:
            param_dist = {
                'n_estimators': randint(200, 1000),
                'max_depth': [10, 20, 50, None],
                'min_samples_split': randint(2, 10)
            }
            search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42, n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = search.best_params_
    else:
        raise ValueError("Tipe masalah tidak didukung.")

    if not perform_tuning:
        model.fit(X_train, y_train)

    return model, X_test, y_test, model_name, best_params