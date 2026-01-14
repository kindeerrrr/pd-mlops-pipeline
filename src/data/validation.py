import pandera as pa
from pandera import Column, DataFrameSchema
import pandas as pd

schema = DataFrameSchema({
    "limit_bal": Column(float, pa.Check.ge(0)),
    "sex": Column(int, pa.Check.isin([1, 2])),
    "education": Column(int),
    "marriage": Column(int),
    "age": Column(int, pa.Check.between(18, 100)),
    "default.payment.next.month": Column(int, pa.Check.isin([0, 1]))
})

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    return schema.validate(df)