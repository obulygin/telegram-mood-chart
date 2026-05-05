"""
Stats Engine: Бутстрэппинг, доверительные интервалы и семантическое взвешивание.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

def bootstrap_confidence_intervals(
    data: pd.DataFrame, 
    value_col: str, 
    n_iterations: int = 1000, 
    confidence_level: float = 0.95,
    show_progress: bool = False
) -> pd.DataFrame:
    """
    Вычисляет доверительные интервалы методом бутстрэппинга для временного ряда.
    
    Параметры:
    - data: DataFrame с колонкой даты (индекс) и значением.
    - value_col: Имя колонки со значениями.
    - n_iterations: Количество итераций бутстрэпа.
    - confidence_level: Уровень доверия (по умолчанию 95%).
    
    Возвращает:
    - DataFrame с колонками: mean, ci_lower, ci_upper, std_err.
    """
    if len(data) == 0:
        return pd.DataFrame(columns=['mean', 'ci_lower', 'ci_upper', 'std_err'])

    dates = data.index
    values = data[value_col].values
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    results = []
    
    iterator = tqdm(dates, desc="Bootstrapping CI") if show_progress else dates
    
    for date in iterator:
        # Получаем все сообщения за эту дату (предполагаем, что на вход подается агрегированный датафрейм,
        # но для бутстрэппинга нам нужна исходная дисперсия. 
        # В данном контексте мы эмулируем дисперсию, если на входе уже агрегат, 
        # НО правильнее передавать сюда сырые данные по дням.
        # Для совместимости с пайплайном: предполагаем, что 'data' здесь - это сырые данные за день,
        # либо мы применяем бутстрэп к скользящему окну, если данные уже агрегированы.
        
        # Корректная логика для этого проекта:
        # Функция вызывается внутри агрегации по дням. 
        # Здесь мы реализуем утилиту для расчета статистик по массиву значений.
        pass

    # Переписываем логику для поэлементной обработки в цикле агрегации
    # Эта функция будет вспомогательной для одной точки данных (списка значений за день)
    raise NotImplementedError("Используйте calculate_daily_stats для подневного расчета.")


def calculate_daily_stats(values: List[float], n_iterations: int = 1000) -> Dict[str, float]:
    """
    Рассчитывает статистику для одного дня (списка значений метрики).
    Возвращает среднее, нижнюю и верхнюю границы CI, стандартную ошибку.
    """
    if not values:
        return {'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'std_err': np.nan}
    
    arr = np.array(values)
    mean_val = np.mean(arr)
    
    if len(arr) < 2:
        return {'mean': mean_val, 'ci_lower': mean_val, 'ci_upper': mean_val, 'std_err': 0.0}
    
    # Бутстрэппинг
    bootstrap_means = []
    for _ in range(n_iterations):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    std_err = np.std(bootstrap_means)
    
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    
    return {
        'mean': mean_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std_err': std_err
    }


def semantic_weighting(embeddings: np.ndarray, neutral_vector: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Вычисляет веса сообщений на основе их удаленности от нейтральной точки.
    Чем дальше вектор от нейтрали, тем выше вес (более эмоционально насыщенное сообщение).
    
    Параметры:
    - embeddings: Матрица эмбеддингов сообщений (N, D).
    - neutral_vector: Вектор нейтрального смысла. Если None, используется нулевой вектор или центроид.
    
    Возвращает:
    - Массив весов (N,).
    """
    if neutral_vector is None:
        # Используем центроид всех сообщений как "норму" или просто нулевой вектор, если эмбеддинги нормализованы
        # Для sentence-transformers лучше использовать вектор "ничего" или среднее по корпусу.
        # Упрощенно: расстояние от центра облака точек.
        neutral_vector = np.mean(embeddings, axis=0)
    
    # Косинусное расстояние или Евклидово? Для весов лучше Евклидово в нормализованном пространстве
    # или 1 - косинусное сходство.
    # Нормализуем эмбеддинги
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normalized_embeddings = embeddings / norms
    
    norm_neutral = np.linalg.norm(neutral_vector)
    if norm_neutral > 0:
        neutral_vector = neutral_vector / norm_neutral
    
    # Косинусное сходство с нейтралью
    similarity = np.dot(normalized_embeddings, neutral_vector)
    
    # Вес: чем меньше сходство с нейтралью, тем выше вес.
    # Диапазон similarity [-1, 1]. 
    # distance = 1 - similarity. 
    weights = 1 - similarity
    
    # Нормализуем веса к диапазону [0.1, 2.0], чтобы не обнулять важные сообщения совсем
    min_w, max_w = weights.min(), weights.max()
    if max_w > min_w:
        weights = 0.1 + (weights - min_w) * (1.9 / (max_w - min_w))
    else:
        weights = np.ones_like(weights)
        
    return weights


def granger_causality_test(data: pd.DataFrame, col_target: str, col_source: str, max_lag: int = 5) -> Dict:
    """
    Упрощенная реализация теста Грейнджера на причинность.
    Проверяет, улучшает ли знание прошлых значений col_source прогноз col_target.
    
    Возвращает словарь с p-value для каждого лага.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return {"error": "statsmodels not installed", "p_values": {}}

    if len(data) < max_lag + 10:
        return {"error": "Not enough data", "p_values": {}}
    
    subset = data[[col_target, col_source]].dropna()
    if len(subset) < max_lag + 10:
        return {"error": "Not enough data after dropna", "p_values": {}}

    try:
        # Тест возвращает таблицу, нам нужен p-value F-теста для максимального лага
        result = grangercausalitytests(subset, max_lag=max_lag, verbose=False)
        p_values = {}
        for lag, tests in result.items():
            # tests[0] содержит p-value F-теста
            p_values[lag] = tests[0]['ssr_ftest'][1]
        return {"p_values": p_values, "significant": any(p < 0.05 for p in p_values.values())}
    except Exception as e:
        return {"error": str(e), "p_values": {}}


def cluster_emotional_states(data: pd.DataFrame, n_clusters: int = 4) -> pd.Series:
    """
    Кластеризует эмоциональные состояния на основе основных метрик.
    Использует K-Means.
    
    Возвращает серию с метками кластеров.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    metrics_cols = [c for c in data.columns if c not in ['date', 'ci_lower', 'ci_upper']]
    # Берем только числовые колонки метрик
    metrics_cols = [c for c in metrics_cols if data[c].dtype == float and not data[c].isna().all()]
    
    if len(metrics_cols) < 2 or len(data) < n_clusters:
        return pd.Series([0] * len(data), index=data.index)
    
    subset = data[metrics_cols].dropna()
    if len(subset) < n_clusters:
        return pd.Series([0] * len(data), index=data.index)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(subset)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_data)
    
    # Возвращаем полный индекс, заполняя пропуски -1
    full_labels = pd.Series(-1, index=data.index)
    full_labels.loc[subset.index] = labels
    
    return full_labels
