import pandas as pd

def predict_new_data(new_data: dict, model, preprocessor):
    try:
        new_df = pd.DataFrame([new_data])
        
        new_data_processed = preprocessor.transform(new_df)
        
        prediction = model.predict(new_data_processed)
        
        return {"prediction": prediction[0].tolist()}

    except Exception as e:
        return {"error": f"Gagal melakukan prediksi. Pastikan data baru memiliki semua kolom yang dibutuhkan. Detail: {str(e)}"}