import pandas as pd
from typing import Literal

def detect_problem_type(target_column: pd.Series) -> Literal["Klasifikasi", "Regresi"]:
    target_column = target_column.dropna()

    if pd.api.types.is_object_dtype(target_column):
        return "Klasifikasi"

    if pd.api.types.is_numeric_dtype(target_column):
        unique_values = target_column.nunique()

        if unique_values == 2:
            return "Klasifikasi"
        # --------------------------------

        if unique_values <= 15 and pd.api.types.is_integer_dtype(target_column):
            return "Klasifikasi"
        else:
            return "Regresi"
    
    return "Klasifikasi"