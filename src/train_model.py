import click
import pandas as pd
from pathlib import Path
import numpy as np
np.random.seed(42)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, r2_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
#import tensorflow as tf
#from keras.models import Model
#from keras.layers import Dense
import matplotlib.pyplot as plt
import json
from utils import w_pickle
import pickle
from settings import Settings
from app_settings import AppSettings
settings = Settings()
app_settings = AppSettings()


def train_model(df: pd.DataFrame,
                model: RandomForestClassifier
                ):
    """
    Служебная функция. выполняет:
        - разделение на Train и test
        - обучение модели Бинарной классификации по типу вина 
        df- датафрейм, random_seed - фиксатор генератора случ.чисел, 
        model - модель классификатора,
        - сохранение модели
    """
    X = df.drop(columns=['type'])
    y = df['type']
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2,
                         random_state=settings.RANDOM_SEED)

    rf = model
    rf.fit(X_train, y_train)
    w_pickle(app_settings.TYPE_MODEL_FILE, rf)
    print('binary model fitted')
    return X, y, X_test, y_test, rf


def get_labels(data: pd.DataFrame):
    """ Получаем метки типа и качества из всего набора данных
    Args: pd.DataFrame
    Return: a tuple of 2 np.ndarrays"""
    type_wine = data.pop('type')
    type_wine = np.array(type_wine)
    return (type_wine)

def make_model():
    """ Функция формирует модель бинарной классификации"""
    
    model = RandomForestClassifier (criterion='entropy',
                            random_state=settings.RANDOM_SEED)
    print('binary model builded')
    return model
   
    
def eval_model(X, y, X_test, y_test, model, flag):
    """
    Служебная функция. выполняет:
        - расчет метрик качества модели бинарной классификации вина по типу в виде classification_report
        - построение confusion matrix
        - кросс-валидацию модели
        args:
        X: pd.DataFrame - датафрейм признаков,
        y: np.ndarray - массив целевой переменной,
        X_test: pd.DataFrame - тестовый датафрейм
        y_test: np.ndarray - тестовый массив целевой переменной,
        random_seed - фиксатор генератора случ.чисел, 
        model - модель классификатора,
        flag - признак печати результатов (True/False)
    """
    num_folds = 9
    scoring = 'r2'

    kfold = StratifiedKFold(n_splits=num_folds,
                            random_state=settings.RANDOM_SEED,
                            shuffle=True)

    y_pred = model.predict(X_test)
    target_names = ['red', 'white']
    # Отчет полностью
    report = classification_report(y_test, y_pred,
                                   target_names=target_names, output_dict=True)
    scoring = ['precision_macro', 'recall_macro']
    cv_results = cross_validate(model, X, y, cv=kfold, scoring=scoring)
    print(cv_results)
    w_pickle(Path('reports', 'cv_results'), cv_results.pkl)

    if flag:
        print("Classification report\n")
        print(report)
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 3))

    cm = confusion_matrix(y_test, y_pred)
    cmp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    cmp.plot(ax=ax, xticks_rotation='vertical')
    fig.savefig(str(Path('reports', "Confusion matrix.png")))

    return cv_results


@click.command()
@click.option('--stage_', help="""Для запуска функций из этого модуля.
--make_model - to build & compile model,
--train_model - fit binary classification model,
--eval_model - to evaluate binary classification model""")
def main(stage_: str) -> None:
    if stage_ == 'make_model':
        _ = make_model()
    
    elif stage_ == 'train_model':
        df = pd.read_csv(app_settings.PROCESSED_TYPE_DATA_FILE,
                         index_col=False)
        X, y, X_test, y_test, rf = train_model(df,
                                               train_model())
        
    elif stage_ == 'eval_model':
        df = pd.read_csv(app_settings.PROCESSED_TYPE_DATA_FILE,
                         index_col=False)
        rf = train_model()
        X, y, X_test, y_test, rf = train_model(df, rf)
        _ = eval_model(X, y, X_test, y_test, rf,
                       flag=True)

if __name__ == '__main__':

    main()
    
    
    #@click.command()
#@click.option('--type_', help="binary or cnn")
#@click.option('--stage_', help="""To run functions from this module.
#--make_model - to build & compile cnn model,
#--train_model - fit binary classification model,
#--train_model_cnn - fit cnn model,
#--eval_model_cnn - to evaluate multy-input cnn model,
#--eval_model - to evaluate binary classification model""")
#def main(stage_: str, type_: str) -> None:
    #if stage_ == 'make_model':
        #_ = make_model(type_)

    #elif stage_ == 'train_model':
        #df = pd.read_csv(app_settings.PROCESSED_TYPE_DATA_FILE,
                         #index_col=False)
        #X, y, X_test, y_test, rf = train_model(df,
                                               #make_model(type_))
    #elif stage_ == 'eval_model':
        #df = pd.read_csv(app_settings.PROCESSED_TYPE_DATA_FILE,
                         #index_col=False)
        #rf = make_model(type_)
        #X, y, X_test, y_test, rf = train_model(df, rf)
        #_ = eval_model(X, y, X_test, y_test, rf,
                       #flag=True)

    #elif stage_ == 'train_model_cnn':
        #df = pd.read_csv(app_settings.PROCESSED_QUALITY_DATA_FILE,
                         #index_col=False)
        #_ = train_model_cnn(df, 'cnn')

    #elif stage_ == 'train_model_transform':
        #df = pd.read_csv(app_settings.PROCESSED_QUALITY_TRANSFORMED_DATA_FILE,
                         #index_col=False)
        #_ = train_model_cnn(df, 'transform')


#if __name__ == '__main__':

    #main()