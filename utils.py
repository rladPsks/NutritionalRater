import pandas as pd

def read_csv(file_name):
    # Read the CSV indexing by barcode strings
    df = pd.read_csv(file_name, dtype={"code": "string"})
    # Index by code without droping the column in the CSV
    df = df.set_index("code", drop=False)
    # Make sure that there are no duplicate codes
    assert df.index.is_unique, "Duplicate barcodes found in the dataset"
    return df