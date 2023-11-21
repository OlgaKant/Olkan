#import click
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import QuantileTransformer
from settings import Settings
from app_settings import AppSettings
settings = Settings()
app_settings = AppSettings()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Конструирование признаков
    Args: pd.DataFrame
    Return: pd.DataFrame"""
    # Замена категориальной переменной type (тип вина) на числовую (0 и 1)
    replace_dict = {'red': 0, 'white': 1}
    df['type'] = df['type'].map(replace_dict)

    # определяем колонки с числовыми признаками
    numeric_features =\
        df.drop(columns=['type', 'quality']).columns.tolist()

    # Подготовка датафрейма для бинарной классификации вин по типу
    X = df[numeric_features]

    y = df['type']
    X_reg = pd.concat([X, y], axis=1)

    sm = SMOTE(random_state=settings.RANDOM_STATE, k_neighbors=4)
    X, y = sm.fit_resample(X, y)
    y_tmp = y.to_frame()
    data = pd.concat([X, y_tmp], axis=1)  # датафрейм после оверсэмплинга.
    data.to_csv(app_settings.PROCESSED_TYPE_DATA_FILE,
                index=False, columns=data.columns)
    return data


def main():
    df = pd.read_csv('data/interim/cleaned_wines.csv', index_col=False)
    _ = build_features(df)
    
if __name__ == '__main__':
    main()
     
    
