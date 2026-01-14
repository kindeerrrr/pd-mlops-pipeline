import pandas as pd
import pytest
from src.data.validation import validate_data

def test_validation_passes():
    df = pd.read_csv("data/processed/clean_data.csv")
    validated = validate_data(df)
    assert validated.shape[0] > 0