"""
LLM Analyzer: Контекстуальный анализ настроений на основе трансформеров.
Использует sentence-transformers для вычисления эмбеддингов и косинусного сходства.
Включает кэширование и батчинг для оптимизации производительности.
"""
import numpy as np
import pandas as pd
import sqlite3
import hashlib
import os
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Глобальные переменные для модели (ленивая загрузка)
_model = None
_tokenizer = None
_device = "cpu"

PROMPTS = {
    "sentiment": [
        "Я чувствую себя прекрасно, мир замечателен",
        "Мне плохо, всё ужасно",
        "Обычный день, ничего особенного"
    ],
    "anxiety": [
        "Я очень тревожусь, мне страшно",
        "Я спокоен и уверен",
        "Немного нервничаю"
    ],
    "stress": [
        "Я под огромным давлением, не справляюсь",
        "У меня всё под контролем",
        "Чувствую напряжение"
    ],
    "i_focus": [
        "Я, мне, моё, я думаю, я чувствую",
        "Мы, нас, вместе, люди, общество",
        "Ты, он, они, друг"
    ],
    "future_orientation": [
        "Завтра, скоро, в будущем, планирую, надеюсь",
        "Вчера, раньше, в прошлом, было",
        "Сейчас, сейчас, в моменте"
    ],
    "social_breadth": [
        "Встреча, группа, вечеринка, много людей, друзья",
        "Один, одиночество, дома, тишина",
        "Пара, несколько человек"
    ],
    "initiative": [
        "Давай встретимся, что думаешь?, как ты?",
        "Нормально, ок, ясно",
        "Не знаю, может быть"
    ],
    "lexical_diversity": [
        "Уникальные редкие сложные слова",
        "Простые обычные слова",
        "Повторяющиеся фразы"
    ],
    "questioning": [
        "Почему? Как? Что? Где? Когда?",
        "Утверждение. Факт. Мнение.",
        "Восклицание! Эмоция!"
    ],
    "negativity_third_party": [
        "Он ужасен, она плохая, они неправы",
        "Он молодец, она умная, они хорошие",
        "Обычный человек"
    ]
}

def get_model():
    """Ленивая загрузка модели."""
    global _model, _tokenizer
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Используем легкую мультитязычную модель
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            print(f"Loading model: {model_name}...")
            _model = SentenceTransformer(model_name)
            print("Model loaded.")
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
    return _model

def _get_cache_db_path():
    """Путь к базе данных кэша."""
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "llm_embeddings_cache.db")

def _get_text_hash(text: str) -> str:
    """Вычисляет MD5 хэш текста для кэширования."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_cached_embeddings(texts: List[str]) -> Tuple[np.ndarray, List[int]]:
    """
    Получает эмбеддинги из кэша или вычисляет новые.
    Возвращает массив эмбеддингов и список индексов, которые нужно вычислить.
    """
    db_path = _get_cache_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Создаем таблицу если нет
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            hash TEXT PRIMARY KEY,
            embedding BLOB
        )
    """)
    conn.commit()
    
    embeddings = []
    to_compute_indices = []
    
    for i, text in enumerate(texts):
        text_hash = _get_text_hash(text)
        cursor.execute("SELECT embedding FROM embeddings WHERE hash = ?", (text_hash,))
        row = cursor.fetchone()
        
        if row:
            emb = np.frombuffer(row[0], dtype=np.float32)
            embeddings.append(emb)
        else:
            embeddings.append(None)
            to_compute_indices.append(i)
    
    conn.close()
    return np.array(embeddings), to_compute_indices

def save_embeddings_to_cache(texts: List[str], embeddings: np.ndarray, indices: List[int]):
    """Сохраняет новые эмбеддинги в кэш."""
    if not indices:
        return
        
    db_path = _get_cache_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for i, idx in enumerate(indices):
        text = texts[idx]
        emb = embeddings[i]
        text_hash = _get_text_hash(text)
        
        cursor.execute(
            "INSERT OR REPLACE INTO embeddings (hash, embedding) VALUES (?, ?)",
            (text_hash, emb.tobytes())
        )
    
    conn.commit()
    conn.close()

def compute_embeddings_batched(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Вычисляет эмбеддинги батчами с прогресс-баром."""
    model = get_model()
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing Embeddings"):
        batch = texts[i:i+batch_size]
        with torch.no_grad():
            batch_embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(batch_embs)
    
    return np.vstack(all_embeddings)

def calculate_metric_scores(embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Вычисляет scores для всех метрик на основе косинусного сходства с промптами.
    """
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    
    model = get_model()
    results = {}
    
    # Энкодим промпты один раз
    prompt_embeddings = {}
    for metric, prompts in PROMPTS.items():
        # Энкодим промпты
        p_embs = model.encode(prompts, convert_to_numpy=True)
        prompt_embeddings[metric] = p_embs
    
    for metric, p_embs in prompt_embeddings.items():
        if len(p_embs) >= 2:
            # Positive vs Negative logic
            # Для большинства метрик: индекс 0 - высокий уровень признака, 1 - низкий
            # sentiment: 0=позитив, 1=негатив -> score = sim(0) - sim(1)
            # anxiety: 0=тревога, 1=спокойствие -> score = sim(0) - sim(1)
            
            pos_emb = p_embs[0].reshape(1, -1)
            neg_emb = p_embs[1].reshape(1, -1)
            
            sim_pos = cosine_similarity(embeddings, pos_emb).flatten()
            sim_neg = cosine_similarity(embeddings, neg_emb).flatten()
            
            # Нормализуем к диапазону [-1, 1] или [0, 1]
            # Разность сходств
            scores = sim_pos - sim_neg
            
            # Если есть нейтральный промпт (индекс 2), можно использовать его для калибровки
            if len(p_embs) > 2:
                neu_emb = p_embs[2].reshape(1, -1)
                sim_neu = cosine_similarity(embeddings, neu_emb).flatten()
                # Упрощенно: просто разность позитива и негатива
                # Можно добавить сложную логику с нейтралью
            
            results[metric] = scores
        else:
            # Fallback
            results[metric] = np.zeros(len(embeddings))
            
    return results

def analyze_messages_llm(messages: List[str]) -> pd.DataFrame:
    """
    Основной пайплайн анализа сообщений через LLM.
    Возвращает DataFrame с метриками для каждого сообщения.
    """
    if not messages:
        return pd.DataFrame()
    
    # 1. Получаем кэш
    cached_embs, to_compute_idx = get_cached_embeddings(messages)
    
    # 2. Вычисляем недостающие
    if to_compute_idx:
        texts_to_compute = [messages[i] for i in to_compute_idx]
        new_embs = compute_embeddings_batched(texts_to_compute)
        
        # Сохраняем в кэш
        save_embeddings_to_cache(messages, new_embs, to_compute_idx)
        
        # Вставляем в общий массив
        for i, idx in enumerate(to_compute_idx):
            cached_embs[idx] = new_embs[i]
    
    embeddings = cached_embs
    
    # 3. Считаем метрики
    scores = calculate_metric_scores(embeddings)
    
    # 4. Семантическое взвешивание
    try:
        from stats_engine import semantic_weighting
        weights = semantic_weighting(embeddings)
    except ImportError:
        weights = np.ones(len(messages))
    
    # 5. Собираем DataFrame
    df = pd.DataFrame(scores)
    df['semantic_weight'] = weights
    df['text'] = messages
    
    return df

if __name__ == "__main__":
    # Тест
    test_msgs = ["Я счастлив!", "Мне грустно", "Все нормально", "Я очень тревожусь"]
    df = analyze_messages_llm(test_msgs)
    print(df.head())
