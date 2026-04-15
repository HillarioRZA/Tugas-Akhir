from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import pandas as pd

def evaluate_model(model, X_test, y_test, problem_type: str) -> dict:
    y_pred = model.predict(X_test)

    if problem_type == "Klasifikasi":
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
    elif problem_type == "Regresi":
        metrics = {
            "mean_squared_error": mean_squared_error(y_test, y_pred),
            "mean_absolute_error": mean_absolute_error(y_test, y_pred),
            "r2_score": r2_score(y_test, y_pred)
        }
    else:
        return {"error": "Tipe masalah tidak didukung."}

    return metrics

def get_feature_importance(model, preprocessor) -> list[dict]:
    try:
        feature_names = preprocessor.get_feature_names_out()

        importances = model.feature_importances_

        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        top_10_features = feature_importance_df.head(10)
        
        return top_10_features.to_dict('records')
        
    except AttributeError:
        return [{"error": "Model yang digunakan tidak mendukung ekstraksi kepentingan fitur."}]
    except Exception as e:
        return [{"error": f"Terjadi kesalahan: {str(e)}"}]
