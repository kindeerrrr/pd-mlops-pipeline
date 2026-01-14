import pandas as pd
import pytest
from src.data.validation import validate_data


def test_validation_passes():
    df = pd.DataFrame({
        "limit_bal": [20000],
        "sex": [1],
        "education": [2],
        "marriage": [1],
        "age": [30],
        "default.payment.next.month": [0]
    })
    validate_data(df)


def test_validation_fails():
    df = pd.DataFrame({
        "limit_bal": [-100],
        "sex": [3],
        "education": [2],
        "marriage": [1],
        "age": [10],
        "default.payment.next.month": [5]
    })
    with pytest.raises(Exception):
        validate_data(df)
