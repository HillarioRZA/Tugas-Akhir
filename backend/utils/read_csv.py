import pandas as pd
import io

def _read_csv_with_fallback(file_contents: bytes):
    try:
        return pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
    except UnicodeDecodeError:
        print("UTF-8 decoding failed, falling back to latin-1.")
        return pd.read_csv(io.StringIO(file_contents.decode('latin-1')))
    except Exception as e:
        print(f"Failed to read CSV with all encodings: {e}")
        return None