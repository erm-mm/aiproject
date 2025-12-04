import numpy as np
import pytest

# Импортируем регистрированные метрики, чтобы registry был заполнен
from a4s_eval.metrics.prediction_metrics import regression_metrics

# Импорт функции run для реального датасета
from tests.additional_files.train_regression_model import run
from a4s_eval.metric_registries.prediction_metric_registry import prediction_metric_registry


def test_regression_metrics_on_real_data():
    # Загружаем данные и обучаем модель
    y_test, y_pred = run()

    # Проверяем, что данные корректного типа
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert len(y_test) == len(y_pred)

    # Берем зарегистрированные функции метрик из registry
    mae_func = prediction_metric_registry.get_functions()["mae"]
    mse_func = prediction_metric_registry.get_functions()["mse"]
    r2_func = prediction_metric_registry.get_functions()["r2"]

    # Создаем простой объект Dataset с полем y для теста метрик
    DummyDataset = type("Dataset", (), {"y": y_test})()
    
    # Вычисляем метрики
    mae_result = mae_func(None, None, DummyDataset, y_pred)[0]
    mse_result = mse_func(None, None, DummyDataset, y_pred)[0]
    r2_result = r2_func(None, None, DummyDataset, y_pred)[0]

    # Проверяем, что значения метрик — числа
    assert isinstance(mae_result, float)
    assert isinstance(mse_result, float)
    assert isinstance(r2_result, float)

    # Выводим результаты для наглядности
    print("\nMAE:", mae_result)
    print("MSE:", mse_result)
    print("R2:", r2_result)
