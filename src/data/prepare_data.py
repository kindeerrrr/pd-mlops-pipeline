import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/default.csv"
PROCESSED_PATH = "data/processed/clean_data.csv"

def prepare_data():
    df = pd.read_csv(RAW_PATH)

    # Переименуем таргет для удобства
    df = df.rename(columns={"default.payment.next.month": "target"})

    # Убираем пропуски (для учебного проекта — ок)
    df = df.dropna()

    # Сохраняем подготовленные данные
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved processed data to {PROCESSED_PATH}")

if __name__ == "__main__":
    prepare_data()
