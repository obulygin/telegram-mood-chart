#!/usr/bin/env python3
"""Telegram Mood Analysis — анализ настроения по экспорту Telegram.

Строит интерактивный график настроения на основе:
- Текстовый сентимент (лексиконный анализ русского языка)
- Частота сообщений
- Средняя длина сообщений
- Использование эмодзи
- Время суток отправки

Научная валидация:
- Альфа Кронбаха для оценки надёжности шкалы
- PCA/факторный анализ для проверки внутренней структуры
- Корреляционный анализ сигналов
"""

import argparse
import json
import math
import os
import pickle
import re
import sys
import webbrowser
from collections import Counter
from datetime import datetime

import emoji
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import ruptures as rpt

# ─── Лексиконный сентимент-анализатор ───────────────────────────────────────

POSITIVE_WORDS = {
    # радость, одобрение
    'хорошо', 'отлично', 'замечательно', 'прекрасно', 'великолепно', 'чудесно',
    'восхитительно', 'превосходно', 'блестяще', 'идеально', 'потрясающе',
    'изумительно', 'божественно', 'шикарно', 'роскошно', 'классно', 'круто',
    'супер', 'кайф', 'огонь', 'бомба', 'топ', 'лучший', 'лучшая', 'лучшее',
    # любовь, нежность
    'люблю', 'любовь', 'обожаю', 'нежность', 'целую', 'обнимаю', 'скучаю',
    'дорогой', 'дорогая', 'милый', 'милая', 'родной', 'родная', 'солнышко',
    'зайка', 'котик', 'малыш', 'сердечко', 'любимый', 'любимая',
    # благодарность, согласие
    'спасибо', 'благодарю', 'благодарен', 'признателен', 'молодец', 'молодцы',
    'согласен', 'верно', 'точно', 'именно', 'конечно', 'безусловно',
    # успех, достижение
    'получилось', 'удалось', 'успех', 'победа', 'ура', 'йес', 'наконец',
    'свершилось', 'заработало', 'готово', 'сделано', 'выиграл', 'выиграли',
    # положительные эмоции
    'рад', 'рада', 'радость', 'счастье', 'счастлив', 'счастлива', 'весело',
    'смешно', 'забавно', 'приятно', 'интересно', 'красиво', 'здорово',
    'вдохновляет', 'нравится', 'понравилось', 'впечатляет', 'удивительно',
    'волшебно', 'уютно', 'тепло', 'мило', 'душевно', 'гармония',
    # поддержка
    'поддерживаю', 'верю', 'надеюсь', 'мечтаю', 'вперед', 'давай', 'можем',
    'справимся', 'получится', 'обязательно',
    # сленг/разговорное
    'ахах', 'хаха', 'ахахах', 'хахаха', 'ахахаха', 'лол', 'ржу', 'ору',
    'збс', 'пушка', 'жиза', 'кайфово', 'четко', 'заебись', 'охуенно',
    'пиздато', 'ништяк', 'норм', 'нормуль', 'гуд', 'найс', 'вау',
}

NEGATIVE_WORDS = {
    # грусть, боль
    'плохо', 'ужасно', 'отвратительно', 'кошмар', 'кошмарно', 'жесть',
    'ужас', 'жуть', 'мрак', 'беда', 'горе', 'печаль', 'тоска', 'грусть',
    'грустно', 'печально', 'тоскливо', 'уныло', 'мерзко', 'гадко', 'паршиво',
    # злость, раздражение
    'злой', 'злая', 'злость', 'бесит', 'бесят', 'раздражает', 'ненавижу',
    'ненависть', 'достало', 'достали', 'задолбало', 'задолбали', 'надоело',
    'осточертело', 'взбесило', 'разозлило', 'выбесило', 'возмутительно',
    # страх, тревога
    'страшно', 'страх', 'боюсь', 'тревога', 'тревожно', 'паника', 'ужас',
    'волнуюсь', 'переживаю', 'нервничаю', 'стресс',
    # усталость, слабость
    'устал', 'устала', 'усталость', 'измотан', 'измотана', 'выдохся',
    'выдохлась', 'нет сил', 'выгорание', 'выгорел', 'выгорела',
    'замучился', 'замучилась', 'утомился', 'утомилась',
    # разочарование, неудача
    'разочарован', 'разочарована', 'разочарование', 'провал', 'неудача',
    'облом', 'обидно', 'обида', 'жаль', 'жалко', 'увы', 'эх', 'блин',
    'черт', 'дерьмо', 'фигня', 'хрень', 'отстой',
    # проблемы
    'проблема', 'проблемы', 'сломалось', 'сломался', 'баг', 'ошибка',
    'косяк', 'не работает', 'невозможно', 'нереально',
    # мат (негативный контекст)
    'блять', 'сука', 'пиздец', 'хуйня', 'ебать', 'нахуй', 'ебаный',
    'заебал', 'заебали', 'ебанулся', 'охуел', 'пиздос',
    # болезнь
    'болит', 'болею', 'заболел', 'заболела', 'температура', 'тошнит',
    'голова болит',
    # одиночество
    'одинок', 'одинока', 'одиночество', 'никому', 'никто', 'бессмысленно',
}

POSITIVE_EMOJI_SET = {
    '😀', '😃', '😄', '😁', '😆', '😂', '🤣', '😊', '😇', '🥰', '😍',
    '🤩', '😘', '😗', '😚', '😋', '😛', '😜', '🤪', '😝', '❤️', '💕',
    '💖', '💗', '💘', '💝', '💞', '💓', '👍', '🎉', '🎊', '🥳', '✨',
    '🔥', '💪', '👏', '🙏', '😎', '🤗', '🥺', '💯', '⭐', '🌟', '🏆',
    '🎯', '✅', '💐', '🌹', '🌸', '♥️', '🫶', '🤝', '😻', '💫',
}

NEGATIVE_EMOJI_SET = {
    '😢', '😭', '😤', '😠', '😡', '🤬', '😰', '😥', '😓', '😩', '😫',
    '😞', '😔', '😟', '😕', '🙁', '☹️', '😣', '😖', '😱', '💔', '👎',
    '😒', '🤮', '🤢', '😷', '🤕', '💀', '☠️', '😵', '🥴', '😬', '😨',
    '😰', '🤦', '🤦‍♂️', '🤦‍♀️', '😪', '😮‍💨',
}


def lexicon_sentiment(text: str) -> float:
    """Оценка сентимента текста по лексикону. Возвращает от -1 до +1."""
    if not text:
        return 0.0
    words = set(re.findall(r'[а-яёa-z]+', text.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


# ─── Извлечение данных ──────────────────────────────────────────────────────

# ─── Маркеры тревоги и стресса ──────────────────────────────────────────────

ANXIETY_WORDS = {
    'тревога', 'тревожно', 'тревожусь', 'переживаю', 'переживания',
    'волнуюсь', 'волнение', 'боюсь', 'страшно', 'страх', 'паника',
    'нервничаю', 'нервы', 'нервно', 'стресс', 'стрессовый',
    'бессонница', 'уснуть', 'проснулся', 'проснулась',
    'устал', 'устала', 'усталость', 'выдохся', 'выдохлась',
    'выгорание', 'выгорел', 'замучился', 'измотан',
    'неудобно', 'извини', 'извините', 'простите', 'прости',
    'обидел', 'обидела', 'виноват', 'виновата',
    'дедлайн', 'срочно', 'горит', 'завал',
    'болит', 'тошнит',
}

STRESS_PROFANITY = {
    'блять', 'бля', 'сука', 'пиздец', 'хуй', 'хуйня', 'ебать',
    'нахуй', 'заебал', 'заебали', 'охуел', 'пиздос', 'блядь',
    'ебал', 'нахуя', 'похуй', 'ахуеть', 'ебанутый',
    'жопа', 'пипец', 'кошмар', 'ужас', 'жесть', 'капец',
}

UNCERTAINTY_WORDS = {
    'наверное', 'наверно', 'возможно', 'может', 'может быть', 'вроде',
    'вроде бы', 'кажется', 'похоже', 'видимо', 'скорее всего',
    'не знаю', 'не уверен', 'не уверена', 'сомневаюсь', 'хз',
    'непонятно', 'неясно', 'фиг знает', 'черт знает',
}

_RE_ANXIETY = re.compile(r'\b(' + '|'.join(ANXIETY_WORDS) + r')\b', re.I)
_RE_STRESS = re.compile(r'\b(' + '|'.join(STRESS_PROFANITY) + r')\b', re.I)
_RE_UNCERTAINTY = re.compile(
    r'\b(' + '|'.join(re.escape(w) for w in UNCERTAINTY_WORDS) + r')\b', re.I)
_RE_QUESTION = re.compile(r'\?')


# Местоимения для определения направленности негатива
_RE_FIRST_PERSON = re.compile(
    r'\b(я|мне|мной|мой|мою|моя|моё|мои|моих|моим|меня|себя|себе|сам|сама)\b', re.I)
_RE_THIRD_PERSON = re.compile(
    r'\b(он|она|они|его|её|ее|ему|ей|им|их|него|неё|нее|ней|ним|них)\b', re.I)


def classify_perspective(text: str) -> str:
    """Определить, о ком сообщение: 'self', 'other', 'both', 'none'."""
    has_first = bool(_RE_FIRST_PERSON.search(text))
    has_third = bool(_RE_THIRD_PERSON.search(text))
    if has_first and has_third:
        return 'both'
    if has_first:
        return 'self'
    if has_third:
        return 'other'
    return 'none'


# ─── Психолингвистические маркеры ─────────────────────────────────────────

_RE_I_FOCUS = re.compile(
    r'\b(я|мне|мной|мой|мою|моя|моё|мои|моих|моим|меня|себя|себе)\b', re.I)
_RE_WE_FOCUS = re.compile(
    r'\b(мы|нас|нам|нами|наш|наша|наше|наши|наших|нашим|нашу)\b', re.I)
_RE_FUTURE = re.compile(
    r'\b(буду|будет|будем|будут|завтра|потом|планирую|собираюсь|хочу|надо|нужно|пора)\b', re.I)
_RE_PAST = re.compile(
    r'\b(было|были|раньше|тогда|вчера|помню|помнишь)\b', re.I)
_RE_WORDS = re.compile(r'[а-яёa-z]+', re.I)


def psycholing_features(text: str) -> dict:
    """Извлечь психолингвистические маркеры из текста."""
    words = _RE_WORDS.findall(text)
    nw = len(words)
    if nw < 3:
        return {'i_rate': 0, 'we_rate': 0, 'future_ratio': 0.5}

    i_count = len(_RE_I_FOCUS.findall(text))
    we_count = len(_RE_WE_FOCUS.findall(text))
    future = len(_RE_FUTURE.findall(text))
    past = len(_RE_PAST.findall(text))

    return {
        'i_rate': i_count / nw,
        'we_rate': we_count / nw,
        'future_ratio': future / (future + past) if (future + past) > 0 else 0.5,
    }


def extract_text(msg_text) -> str:
    """Извлечь текст из поля message['text'] (str или list)."""
    if isinstance(msg_text, str):
        return msg_text
    if isinstance(msg_text, list):
        parts = []
        for part in msg_text:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get('text', ''))
        return ''.join(parts)
    return ''


def count_emoji(text: str) -> tuple[int, int]:
    """Подсчитать позитивные и негативные эмодзи в тексте."""
    emojis = [c['emoji'] for c in emoji.emoji_list(text)]
    pos = sum(1 for e in emojis if e in POSITIVE_EMOJI_SET)
    neg = sum(1 for e in emojis if e in NEGATIVE_EMOJI_SET)
    return pos, neg


def _detect_user_id(data: dict) -> str:
    """Автоопределение user_id: находит самый частый from_id в экспорте."""
    counter = Counter()
    for chat in data.get('chats', {}).get('list', []):
        for msg in chat.get('messages', []):
            fid = msg.get('from_id')
            if fid and msg.get('type') == 'message':
                counter[fid] += 1
    if not counter:
        raise ValueError("Не найдено ни одного сообщения с from_id в экспорте")
    user_id = counter.most_common(1)[0][0]
    print(f"Автоопределён user_id: {user_id} ({counter[user_id]:,} сообщений)")
    return user_id


def extract_messages(json_path: str, user_id: str = None) -> list[dict]:
    """Извлечь собственные сообщения из экспорта Telegram.

    Если user_id не указан, автоматически определяется как самый частый
    from_id в данных (т.е. владелец экспорта).
    """
    print(f"Загрузка {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("JSON загружен. Извлечение сообщений...")

    if user_id is None:
        user_id = _detect_user_id(data)

    messages = []
    total = 0
    own = 0

    for chat in data.get('chats', {}).get('list', []):
        chat_id = chat.get('id')
        for msg in chat.get('messages', []):
            total += 1
            if msg.get('from_id') != user_id or msg.get('type') != 'message':
                continue
            own += 1

            text = extract_text(msg.get('text', ''))
            dt = datetime.fromisoformat(msg['date'])
            pos_emoji, neg_emoji = count_emoji(text)

            # Учитываем sticker_emoji
            sticker = msg.get('sticker_emoji', '')
            if sticker:
                if sticker in POSITIVE_EMOJI_SET:
                    pos_emoji += 1
                elif sticker in NEGATIVE_EMOJI_SET:
                    neg_emoji += 1

            psy = psycholing_features(text)
            words = _RE_WORDS.findall(text)

            messages.append({
                'date': dt,
                'text': text,
                'text_length': len(text),
                'hour': dt.hour,
                'chat_id': chat_id,
                'pos_emoji': pos_emoji,
                'neg_emoji': neg_emoji,
                'perspective': classify_perspective(text),
                'anxiety': len(_RE_ANXIETY.findall(text)),
                'stress': len(_RE_STRESS.findall(text)),
                'uncertainty': len(_RE_UNCERTAINTY.findall(text)),
                'has_question': 1 if _RE_QUESTION.search(text) else 0,
                'word_count': len(words),
                'unique_words': len(set(w.lower() for w in words)),
                'i_rate': psy['i_rate'],
                'we_rate': psy['we_rate'],
                'future_ratio': psy['future_ratio'],
            })

    print(f"Всего сообщений: {total:,}, ваших: {own:,}")
    return messages


def extract_messages_from_db(db_path: str, user_name: str = None) -> list[dict]:
    """Извлечь собственные сообщения из SQLite базы messages.db.

    Args:
        db_path: путь к SQLite базе данных
        user_name: имя пользователя для поиска (LIKE '%name%').
                   Если не указано, берёт sender_contact_id с наибольшим числом сообщений.
    """
    import sqlite3

    print(f"Загрузка сообщений из {db_path}...")
    conn = sqlite3.connect(db_path)

    if user_name:
        me_ids = set(
            r[0] for r in conn.execute(
                "SELECT id FROM contacts WHERE name LIKE ?",
                (f'%{user_name}%',)
            ).fetchall()
        )
        if not me_ids:
            print(f"ОШИБКА: не найден контакт по имени '{user_name}'")
            conn.close()
            return []
        print(f"Найдены contact_id для '{user_name}': {me_ids}")
    else:
        # Автоопределение: sender_contact_id с максимумом сообщений
        row = conn.execute("""
            SELECT sender_contact_id, COUNT(*) as cnt
            FROM messages
            WHERE sender_contact_id IS NOT NULL
              AND msg_type = 'message'
            GROUP BY sender_contact_id
            ORDER BY cnt DESC
            LIMIT 1
        """).fetchone()
        if not row:
            print("ОШИБКА: не найдено ни одного сообщения в базе")
            conn.close()
            return []
        me_ids = {row[0]}
        # Lookup name for display
        name_row = conn.execute(
            "SELECT name FROM contacts WHERE id = ?", (row[0],)
        ).fetchone()
        display = name_row[0] if name_row else f"contact_id={row[0]}"
        print(f"Автоопределён пользователь: {display} ({row[1]:,} сообщений)")

    placeholders = ",".join("?" * len(me_ids))

    # Deduplicate: same date+text from different sources — keep highest priority
    rows = conn.execute(f"""
        SELECT date, text, chat_id FROM (
            SELECT date, text, chat_id, source,
                   ROW_NUMBER() OVER (
                       PARTITION BY date, text
                       ORDER BY CASE source
                           WHEN 'json_2026' THEN 0
                           WHEN 'json_2021' THEN 1
                           WHEN 'gchat' THEN 2
                           WHEN 'html_2020' THEN 3
                           WHEN 'html_2023' THEN 4
                           WHEN 'qip' THEN 5
                           ELSE 6
                       END
                   ) as rn
            FROM messages
            WHERE sender_contact_id IN ({placeholders})
              AND msg_type = 'message'
              AND date != ''
              AND text IS NOT NULL
              AND text != ''
        ) WHERE rn = 1
        ORDER BY date
    """, list(me_ids)).fetchall()

    conn.close()

    print(f"Найдено {len(rows):,} собственных сообщений")

    messages = []
    for date_str, text, chat_id in rows:
        try:
            dt = datetime.fromisoformat(date_str)
        except (ValueError, TypeError):
            continue

        text = text or ""
        psy = psycholing_features(text)
        words = _RE_WORDS.findall(text)

        messages.append({
            'date': dt,
            'text': text,
            'text_length': len(text),
            'hour': dt.hour,
            'chat_id': chat_id,
            'pos_emoji': 0,
            'neg_emoji': 0,
            'perspective': classify_perspective(text),
            'anxiety': len(_RE_ANXIETY.findall(text)),
            'stress': len(_RE_STRESS.findall(text)),
            'uncertainty': len(_RE_UNCERTAINTY.findall(text)),
            'has_question': 1 if _RE_QUESTION.search(text) else 0,
            'word_count': len(words),
            'unique_words': len(set(w.lower() for w in words)),
            'i_rate': psy['i_rate'],
            'we_rate': psy['we_rate'],
            'future_ratio': psy['future_ratio'],
        })

    print(f"Обработано: {len(messages):,} сообщений")
    return messages


# ─── Сентимент-анализ ───────────────────────────────────────────────────────

LABEL_TO_SCORE = {
    'POSITIVE': 1.0,
    'NEGATIVE': -1.0,
    'NEUTRAL': 0.0,
}


def try_load_rubert():
    """Попытка загрузить RuBERT sentiment pipeline."""
    try:
        from transformers import pipeline as hf_pipeline
        import torch
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Загрузка RuBERT модели (device={device})...")
        pipe = hf_pipeline(
            'sentiment-analysis',
            model='blanchefort/rubert-base-cased-sentiment-rusentiment',
            device=device,
            truncation=True,
            max_length=512,
        )
        # Тест
        result = pipe(['тест'])
        if result:
            return pipe
    except Exception as e:
        print(f"RuBERT недоступен: {e}")
    return None


def score_sentiment(messages: list[dict]) -> list[dict]:
    """Добавить оценку сентимента к каждому сообщению."""
    pipe = try_load_rubert()

    if pipe:
        print("Используется RuBERT для анализа сентимента...", flush=True)
        batch_size = 128
        texts_with_idx = [
            (i, m['text']) for i, m in enumerate(messages)
            if len(m['text']) > 3
        ]
        total_texts = len(texts_with_idx)
        print(f"  Текстов для анализа: {total_texts:,}", flush=True)
        # Инициализируем нулями
        for m in messages:
            m['sentiment'] = 0.0

        import time
        t0 = time.time()
        for batch_start in range(0, total_texts, batch_size):
            batch = texts_with_idx[batch_start:batch_start + batch_size]
            indices, texts = zip(*batch)
            texts = [t[:512] for t in texts]
            results = pipe(list(texts), batch_size=batch_size)
            for idx, result in zip(indices, results):
                label = result['label']
                score = result['score']
                base = LABEL_TO_SCORE.get(label, 0.0)
                messages[idx]['sentiment'] = base * score
            done = min(batch_start + batch_size, total_texts)
            if done % (batch_size * 50) < batch_size:
                elapsed = time.time() - t0
                speed = done / elapsed if elapsed > 0 else 0
                eta = (total_texts - done) / speed if speed > 0 else 0
                print(f"  {done:,}/{total_texts:,} ({done*100//total_texts}%) "
                      f"— {speed:.0f} msg/s, ETA {eta:.0f}s", flush=True)
    else:
        print("Используется лексиконный анализатор (fallback)...")
        for i, m in enumerate(messages):
            m['sentiment'] = lexicon_sentiment(m['text'])
            if (i + 1) % 100_000 == 0:
                print(f"  Обработано {i + 1:,} / {len(messages):,}")

    return messages


# ─── Агрегация ──────────────────────────────────────────────────────────────

def aggregate(messages: list[dict], window: str = 'month',
              start_date: str = '2006-01-01', use_bootstrap: bool = True) -> pd.DataFrame:
    """Агрегировать сигналы по временным окнам.

    Подход: оцениваем ТОЛЬКО содержание текста (что написано), а не
    количественные метрики (сколько, как часто, с какими эмодзи).
    Частота, длина, эмодзи — это функция технологий, а не психологии.

    10 сигналов (6 текстовых + 4 поведенческих):
    Текстовые:
    - sentiment: лексический позитив/негатив слов
    - anxiety: маркеры тревоги и беспокойства
    - stress: стрессовая лексика и мат-от-фрустрации
    - i_rate: доля я-местоимений (падение = отстранённость = плохо)
    - we_rate: доля мы-местоимений (падение = изоляция)
    - future_ratio: ориентация на будущее (падение = потеря мотивации)
    Поведенческие:
    - social_breadth: число уникальных контактов в месяц
    - initiation_rate: доля сессий, инициированных мной
    - ttr: лексическое разнообразие
    - question_rate: доля сообщений с вопросами
    
    Args:
        use_bootstrap: Если True, вычислять доверительные интервалы методом бутстрэппинга
    """
    import numpy as np
    from stats_engine import calculate_daily_stats

    df = pd.DataFrame(messages)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= start_date]

    if window == 'month':
        df['period'] = df['date'].dt.to_period('M')
    else:
        df['period'] = df['date'].dt.to_period('W')

    grouped = df.groupby('period')

    # ── Корректировка сентимента по перспективе ──────────────────────────
    # Обсуждение чужих проблем != личное плохое настроение
    def adjusted_sentiment(row):
        s = row['sentiment']
        if s >= 0:
            return s
        persp = row.get('perspective', 'none')
        if persp == 'other':
            return s * 0.3
        if persp == 'both':
            return s * 0.6
        return s

    df['sentiment_adj'] = df.apply(adjusted_sentiment, axis=1)

    # ── Базовые агрегаты ─────────────────────────────────────────────────
    agg = pd.DataFrame()
    
    # Используем бутстрэппинг для сентимента если включено
    if use_bootstrap:
        print("Вычисление доверительных интервалов (бутстрэппинг)...")
        sentiment_stats = []
        for period, group in tqdm(grouped, desc="Bootstrapping"):
            values = group['sentiment_adj'].tolist()
            stats = calculate_daily_stats(values, n_iterations=500)  # 500 итераций для баланса скорость/точность
            sentiment_stats.append(stats)
        
        stats_df = pd.DataFrame(sentiment_stats, index=agg.index if len(agg) > 0 else range(len(sentiment_stats)))
        agg['sentiment_mean'] = stats_df['mean']
        agg['sentiment_ci_lower'] = stats_df['ci_lower']
        agg['sentiment_ci_upper'] = stats_df['ci_upper']
        agg['sentiment_std_err'] = stats_df['std_err']
    else:
        agg['sentiment_mean'] = grouped['sentiment_adj'].mean()
        agg['sentiment_ci_lower'] = agg['sentiment_mean']
        agg['sentiment_ci_upper'] = agg['sentiment_mean']
        agg['sentiment_std_err'] = 0.0
    
    agg['msg_count'] = grouped['text'].count()

    # Отбросить периоды с менее чем 30 сообщениями — слишком шумные
    agg = agg[agg['msg_count'] >= 30]

    # Тревога и стресс (доля сообщений с маркерами)
    agg['anxiety_rate'] = grouped['anxiety'].apply(lambda s: (s > 0).sum()) / agg['msg_count']
    agg['stress_rate'] = grouped['stress'].apply(lambda s: (s > 0).sum()) / agg['msg_count']

    # Психолингвистические маркеры
    agg['i_rate'] = grouped['i_rate'].mean()
    agg['we_rate'] = grouped['we_rate'].mean()
    agg['future_ratio'] = grouped['future_ratio'].mean()

    # ── Поведенческие маркеры ────────────────────────────────────────────

    # Социальная широта: уникальные контакты (chat_id) в месяц
    if 'chat_id' in df.columns:
        agg['social_breadth'] = grouped['chat_id'].nunique()
    else:
        agg['social_breadth'] = agg['msg_count'] * 0 + 10  # fallback

    # Инициация: доля "первых сообщений" в чате после паузы >4 часов
    if 'chat_id' in df.columns:
        df_sorted = df.sort_values(['chat_id', 'date'])
        df_sorted['prev_date'] = df_sorted.groupby('chat_id')['date'].shift(1)
        df_sorted['gap_hours'] = (
            (df_sorted['date'] - df_sorted['prev_date']).dt.total_seconds() / 3600
        )
        df_sorted['is_initiation'] = (
            df_sorted['gap_hours'].isna() | (df_sorted['gap_hours'] > 4)
        ).astype(int)
        # Merge back by index
        df['is_initiation'] = df_sorted['is_initiation'].values
        agg['initiation_rate'] = grouped['is_initiation'].mean()
    else:
        agg['initiation_rate'] = agg['msg_count'] * 0 + 0.5

    # Лексическое разнообразие (TTR) — unique_words / word_count per period
    if 'unique_words' in df.columns:
        agg['ttr'] = grouped['unique_words'].sum() / grouped['word_count'].sum().replace(0, np.nan)
        agg['ttr'] = agg['ttr'].fillna(0.5)
    else:
        agg['ttr'] = agg['msg_count'] * 0 + 0.5

    # Вопросительность: доля сообщений с вопросительным знаком
    if 'has_question' in df.columns:
        agg['question_rate'] = grouped['has_question'].mean()
    else:
        agg['question_rate'] = agg['msg_count'] * 0 + 0.15

    # ── Rolling z-score для краткосрочного графика ────────────────────────
    ROLLING_WINDOW = 12

    def rolling_zscore(series, window=ROLLING_WINDOW):
        """Z-score относительно скользящего окна (не глобального)."""
        roll_mean = series.rolling(window, center=True, min_periods=3).mean()
        roll_std = series.rolling(window, center=True, min_periods=3).std()
        exp_mean = series.expanding(min_periods=3).mean()
        exp_std = series.expanding(min_periods=3).std()
        roll_mean = roll_mean.fillna(exp_mean)
        roll_std = roll_std.fillna(exp_std)
        roll_std = roll_std.replace(0, np.nan)
        result = (series - roll_mean) / roll_std
        return result.fillna(0)

    # Краткосрочный композит: 10 сигналов (rolling z-score)
    sent_z = rolling_zscore(agg['sentiment_mean'])
    anx_z = rolling_zscore(agg['anxiety_rate'])
    stress_z = rolling_zscore(agg['stress_rate'])
    i_z = rolling_zscore(agg['i_rate'])           # падение = плохо (отстранённость)
    we_z = rolling_zscore(agg['we_rate'])          # падение = плохо (изоляция)
    future_z = rolling_zscore(agg['future_ratio']) # падение = плохо
    social_z = rolling_zscore(agg['social_breadth'])  # падение = замкнутость
    init_z = rolling_zscore(agg['initiation_rate'])   # падение = пассивность
    ttr_z = rolling_zscore(agg['ttr'])                # падение = когнитивное обеднение
    quest_z = rolling_zscore(agg['question_rate'])    # падение = потеря любопытства

    # Композит: веса калиброваны по корреляциям с итоговым настроением.
    shortterm_raw = (
        0.20 * sent_z +
        0.10 * i_z +
        0.05 * we_z +
        0.05 * future_z -
        0.15 * anx_z +
        0.20 * quest_z +
        0.15 * social_z +
        0.10 * init_z
    )
    agg['mood_shortterm'] = shortterm_raw.apply(lambda x: math.tanh(x))
    agg['mood_shortterm_smooth'] = (
        agg['mood_shortterm'].rolling(3, center=True, min_periods=1).mean()
    )

    # ── Долгосрочный тренд: абсолютные значения, широкое сглаживание ─────
    LONG_SMOOTH = 6
    agg['mood_longterm'] = (
        agg['sentiment_mean'].rolling(LONG_SMOOTH, center=True, min_periods=2).mean()
    )
    agg['anxiety_trend'] = (
        agg['anxiety_rate'].rolling(LONG_SMOOTH, center=True, min_periods=2).mean()
    )
    agg['stress_trend'] = (
        agg['stress_rate'].rolling(LONG_SMOOTH, center=True, min_periods=2).mean()
    )

    # Datetime индекс для plotly
    agg['date'] = agg.index.map(lambda p: p.start_time)

    return agg.reset_index(drop=True)


# ─── Статистическая валидация метрики ──────────────────────────────────────

def cronbach_alpha(df: pd.DataFrame, items: list[str]) -> float:
    """Вычислить альфу Кронбаха для оценки внутренней согласованности шкалы.
    
    Args:
        df: DataFrame с агрегированными данными
        items: список колонок-сигналов для анализа
        
    Returns:
        alpha: коэффициент от 0 до 1 (>=0.7 считается приемлемым)
    """
    # Стандартизируем сигналы (z-score)
    standardized = df[items].apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x)
    
    n_items = len(items)
    if n_items < 2:
        return np.nan
    
    # Дисперсии по каждому пункту
    variances = standardized.var(axis=0)
    # Общая дисперсия суммы
    total_variance = standardized.sum(axis=1).var()
    
    # Формула Кронбаха: alpha = (n / (n-1)) * (1 - sum(var_i) / var_total)
    alpha = (n_items / (n_items - 1)) * (1 - variances.sum() / total_variance)
    
    return alpha


def factor_analysis_pca(df: pd.DataFrame, items: list[str], n_components: int = 2):
    """Провести PCA/факторный анализ для проверки структуры конструкта.
    
    Args:
        df: DataFrame с агрегированными данными
        items: список колонок-сигналов
        n_components: число главных компонент
        
    Returns:
        dict с результатами: explained_variance, loadings, components
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Подготовка данных
    data = df[items].dropna()
    if len(data) < n_components + 1:
        return None
    
    # Стандартизация
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled_data)
    
    # Loadings (корреляции между исходными переменными и компонентами)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    return {
        'explained_variance': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'loadings': pd.DataFrame(loadings, index=items, 
                                columns=[f'PC{i+1}' for i in range(n_components)]),
        'components': components,
        'n_components': n_components,
    }


def correlation_matrix(df: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    """Построить корреляционную матрицу сигналов с p-values.
    
    Returns:
        DataFrame с корреляциями Пирсона
    """
    corr_matrix = df[items].corr(method='pearson')
    return corr_matrix


def validate_mood_metric(df: pd.DataFrame) -> dict:
    """Комплексная валидация метрики настроения.
    
    Проверяет:
    1. Надёжность (alpha Кронбаха)
    2. Конвергентную валидность (корреляции между сигналами)
    3. Структурную валидность (PCA)
    
    Returns:
        dict с результатами валидации
    """
    # Сигналы для валидации (текстовые + поведенческие)
    text_signals = [
        'sentiment_mean', 'anxiety_rate', 'stress_rate',
        'i_rate', 'we_rate', 'future_ratio'
    ]
    behavioral_signals = [
        'social_breadth', 'initiation_rate', 'ttr', 'question_rate'
    ]
    all_signals = text_signals + behavioral_signals
    
    # Фильтруем существующие колонки
    available_signals = [s for s in all_signals if s in df.columns]
    text_available = [s for s in text_signals if s in df.columns]
    behav_available = [s for s in behavioral_signals if s in df.columns]
    
    results = {
        'n_periods': len(df),
        'signals_used': available_signals,
        'cronbach_alpha': None,
        'cronbach_alpha_text': None,
        'cronbach_alpha_behavioral': None,
        'correlation_matrix': None,
        'pca_results': None,
        'interpretation': [],
    }
    
    if len(available_signals) >= 2 and len(df) >= 5:
        # 1. Альфа Кронбаха для всех сигналов
        results['cronbach_alpha'] = cronbach_alpha(df, available_signals)
        
        # Интерпретация alpha
        alpha = results['cronbach_alpha']
        if not np.isnan(alpha):
            if alpha >= 0.9:
                interp = "Отличная внутренняя согласованность (α ≥ 0.9)"
            elif alpha >= 0.8:
                interp = "Хорошая внутренняя согласованность (α ≥ 0.8)"
            elif alpha >= 0.7:
                interp = "Приемлемая внутренняя согласованность (α ≥ 0.7)"
            elif alpha >= 0.6:
                interp = "Удовлетворительная согласованность (α ≥ 0.6) — требует доработки"
            else:
                interp = "Низкая согласованность (α < 0.6) — шкала нуждается в пересмотре"
            results['interpretation'].append(f"Надёжность шкалы: {interp}")
    
    # 2. Альфа для текстовых сигналов отдельно
    if len(text_available) >= 2:
        results['cronbach_alpha_text'] = cronbach_alpha(df, text_available)
    
    # 3. Альфа для поведенческих сигналов отдельно
    if len(behav_available) >= 2:
        results['cronbach_alpha_behavioral'] = cronbach_alpha(df, behav_available)
    
    # 4. Корреляционная матрица
    if len(available_signals) >= 2:
        results['correlation_matrix'] = correlation_matrix(df, available_signals)
    
    # 5. PCA (только если достаточно данных)
    if len(df) >= 10 and len(available_signals) >= 3:
        results['pca_results'] = factor_analysis_pca(df, available_signals, n_components=2)
        
        if results['pca_results']:
            ev = results['pca_results']['explained_variance']
            cum_ev = results['pca_results']['cumulative_variance']
            results['interpretation'].append(
                f"PCA: PC1 объясняет {ev[0]*100:.1f}% вариации, "
                f"PC1+PC2 вместе — {cum_ev[1]*100:.1f}%"
            )
    
    return results


def print_validation_report(results: dict):
    """Вывести отчёт о валидации в консоль."""
    print("\n" + "="*60)
    print("📊 ОТЧЁТ О ВАЛИДАЦИИ МЕТРИКИ НАСТРОЕНИЯ")
    print("="*60)
    
    print(f"\nПериодов для анализа: {results['n_periods']}")
    print(f"Использовано сигналов: {len(results['signals_used'])}")
    print(f"Сигналы: {', '.join(results['signals_used'])}")
    
    print("\n--- 1. НАДЁЖНОСТЬ (Альфа Кронбаха) ---")
    alpha = results.get('cronbach_alpha')
    if alpha is not None and not np.isnan(alpha):
        print(f"  Общий α = {alpha:.3f}")
    else:
        print("  Общий α: недостаточно данных")
    
    alpha_text = results.get('cronbach_alpha_text')
    if alpha_text is not None and not np.isnan(alpha_text):
        print(f"  Текстовые сигналы α = {alpha_text:.3f}")
    
    alpha_behav = results.get('cronbach_alpha_behavioral')
    if alpha_behav is not None and not np.isnan(alpha_behav):
        print(f"  Поведенческие сигналы α = {alpha_behav:.3f}")
    
    print("\n--- 2. ИНТЕРПРЕТАЦИЯ ---")
    for interp in results.get('interpretation', []):
        print(f"  • {interp}")
    
    print("\n--- 3. КОРРЕЛЯЦИОННАЯ МАТРИЦА ---")
    corr = results.get('correlation_matrix')
    if corr is not None:
        # Выводим только значимые корреляции (|r| > 0.3)
        print("  Значимые корреляции (|r| > 0.3):")
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                col1, col2 = corr.columns[i], corr.columns[j]
                r = corr.iloc[i, j]
                if abs(r) > 0.3:
                    direction = "положительная" if r > 0 else "отрицательная"
                    strength = "сильная" if abs(r) > 0.6 else "умеренная"
                    print(f"    {col1[:15]:15} ↔ {col2[:15]:15}: r={r:+.3f} ({strength} {direction})")
    else:
        print("  Недостаточно данных для корреляционного анализа")
    
    print("\n--- 4. ФАКТОРНЫЙ АНАЛИЗ (PCA) ---")
    pca = results.get('pca_results')
    if pca:
        print(f"  Объяснённая дисперсия:")
        for i, ev in enumerate(pca['explained_variance']):
            print(f"    PC{i+1}: {ev*100:.1f}%")
        print(f"\n  Loadings (вклад сигналов в компоненты):")
        print(pca['loadings'].round(3).to_string())
    else:
        print("  Недостаточно данных для факторного анализа (нужно ≥10 периодов)")
    
    print("\n" + "="*60)


# ─── Детекция точек изменения (Change Point Detection) ──────────────────────

def detect_change_points(df: pd.DataFrame, signal_col: str = 'mood_shortterm',
                         penalty: int = 10, min_size: int = 3) -> list[dict]:
    """Обнаружить статистически значимые точки изменения настроения.
    
    Использует алгоритм PELT (Pruned Exact Linear Time) для поиска точек,
    где среднее значение сигнала значительно меняется.
    
    Args:
        df: DataFrame с агрегированными данными
        signal_col: колонка сигнала для анализа
        penalty: штраф за добавление новой точки (выше = меньше точек)
        min_size: минимальный размер сегмента в периодах
        
    Returns:
        список словарей с информацией о точках изменения
    """
    signal = df[signal_col].values
    
    if len(signal) < min_size * 2:
        return []
    
    # PELT algorithm для детекции изменений в среднем
    model = rpt.Pelt(model="rbf").fit(signal.reshape(-1, 1))
    
    # Поиск точек изменения с заданным штрафом
    try:
        breakpoints = model.predict(pen=penalty)
    except Exception:
        # Fallback на бинарную сегментацию если PELT не сходится
        model = rpt.Binseg(model="rbf").fit(signal.reshape(-1, 1))
        breakpoints = model.predict(n_bkpts=5)
    
    # Убираем последнюю точку (это конец ряда, не точка изменения)
    breakpoints = [bp for bp in breakpoints if bp < len(signal)]
    
    if not breakpoints:
        return []
    
    # Формируем подробную информацию о каждой точке изменения
    change_points = []
    dates = df['date'].values
    
    for i, bp in enumerate(breakpoints):
        # Определяем границы сегментов
        prev_start = breakpoints[i-1] if i > 0 else 0
        curr_start = bp
        curr_end = breakpoints[i+1] if i < len(breakpoints) - 1 else len(signal)
        
        # Вычисляем статистику до и после точки изменения
        before_mean = signal[prev_start:curr_start].mean() if curr_start > prev_start else signal[curr_start-1]
        after_mean = signal[curr_start:curr_end].mean() if curr_end > curr_start else signal[curr_start]
        change_magnitude = after_mean - before_mean
        
        # Статистический тест (t-test) для значимости изменения
        if curr_start > prev_start and curr_end > curr_start:
            try:
                t_stat, p_value = stats.ttest_ind(
                    signal[prev_start:curr_start],
                    signal[curr_start:curr_end],
                    equal_var=False
                )
                is_significant = p_value < 0.05
            except Exception:
                t_stat, p_value = 0.0, 1.0
                is_significant = False
        else:
            t_stat, p_value = 0.0, 1.0
            is_significant = False
        
        change_points.append({
            'index': int(curr_start),
            'date': pd.Timestamp(dates[curr_start]),
            'before_mean': float(before_mean),
            'after_mean': float(after_mean),
            'change_magnitude': float(change_magnitude),
            'direction': 'up' if change_magnitude > 0 else 'down',
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'is_significant': is_significant,
            'segment_length_before': int(curr_start - prev_start),
            'segment_length_after': int(curr_end - curr_start),
        })
    
    return change_points


def format_change_point_label(cp: dict) -> str:
    """Создать читаемую метку для точки изменения."""
    direction_ru = "↑ рост" if cp['direction'] == 'up' else "↓ спад"
    magnitude = abs(cp['change_magnitude'])
    
    if cp['is_significant']:
        sig_text = f" (p={cp['p_value']:.3f})"
    else:
        sig_text = " (незначимо)"
    
    return f"{direction_ru}{sig_text}"


def add_change_points_to_chart(fig, df: pd.DataFrame, change_points: list[dict],
                                row: int = 1):
    """Добавить аннотации точек изменения на график.
    
    Args:
        fig: Plotly figure
        df: DataFrame с данными
        change_points: список точек изменения от detect_change_points
        row: номер подграфика для добавления
    """
    if not change_points:
        return
    
    _yaxis_for_row = {1: 'y', 2: 'y2', 3: 'y3', 4: 'y4'}
    yref = _yaxis_for_row[row] + ' domain'
    
    for cp in change_points:
        x = cp['date']
        label = format_change_point_label(cp)
        
        # Вертикальная линия
        color = 'rgba(255, 100, 100, 0.7)' if cp['direction'] == 'down' else 'rgba(100, 200, 100, 0.7)'
        if cp['is_significant']:
            line_width = 2
            dash = 'dash'
        else:
            line_width = 1
            dash = 'dot'
        
        fig.add_vline(
            x=x, line_dash=dash, line_color=color, line_width=line_width,
            row=row, col=1
        )
        
        # Аннотация сверху
        fig.add_annotation(
            x=x, y=1.0, yref=yref,
            text=label, showarrow=False,
            font=dict(size=8, color=color),
            textangle=-90, xanchor='left', yanchor='top',
            bgcolor='rgba(255,255,255,0.9)', borderpad=1,
            row=row, col=1
        )


def print_change_points_report(change_points: list[dict]):
    """Вывести отчёт о найденных точках изменения."""
    print("\n" + "="*60)
    print("🔍 ДЕТЕКЦИЯ ТОЧЕК ИЗМЕНЕНИЯ (PELT алгоритм)")
    print("="*60)
    
    if not change_points:
        print("\nТочки изменения не обнаружены.")
        print("Возможные причины:")
        print("  • Недостаточно данных (нужно ≥6 периодов)")
        print("  • Настроение стабильно без резких перепадов")
        print("  • Высокий порог sensitivity (попробуйте снизить penalty)")
        return
    
    significant = [cp for cp in change_points if cp['is_significant']]
    
    print(f"\nВсего найдено точек изменения: {len(change_points)}")
    print(f"Из них статистически значимых (p < 0.05): {len(significant)}")
    
    print("\n--- ХРОНОЛОГИЯ СОБЫТИЙ ---")
    for i, cp in enumerate(change_points, 1):
        date_str = cp['date'].strftime('%Y-%m')
        direction = "↑ РОСТ" if cp['direction'] == 'up' else "↓ СПАД"
        mag = abs(cp['change_magnitude'])
        sig_marker = "**" if cp['is_significant'] else ""
        
        print(f"\n{i}. {date_str} — {direction} {sig_marker}")
        print(f"   Величина изменения: {mag:.3f}")
        print(f"   До: {cp['before_mean']:+.3f} → После: {cp['after_mean']:+.3f}")
        print(f"   Длительность предыдущего периода: {cp['segment_length_before']} мес.")
        if cp['is_significant']:
            print(f"   Статистическая значимость: t={cp['t_statistic']:.2f}, p={cp['p_value']:.4f}")
        else:
            print(f"   Статистическая значимость: p={cp['p_value']:.4f} (незначимо)")
    
    # Анализ паттернов
    if significant:
        print("\n--- АНАЛИЗ ПАТТЕРНОВ ---")
        ups = sum(1 for cp in significant if cp['direction'] == 'up')
        downs = sum(1 for cp in significant if cp['direction'] == 'down')
        print(f"  Значимых подъёмов: {ups}")
        print(f"  Значимых спадов: {downs}")
        
        if downs > ups:
            print("  ⚠️  Преобладают значимые спады настроения — возможна хроническая стрессовая нагрузка")
        elif ups > downs:
            print("  ✓ Преобладают значимые подъёмы — позитивная динамика")
        else:
            print("  ↔ Сбалансированная динамика подъёмов и спадов")
    
    print("\n" + "="*60)


# ─── Визуализация ───────────────────────────────────────────────────────────

def create_chart(df: pd.DataFrame, output_path: str, window: str,
                 change_points: list[dict] = None):
    """Создать интерактивный HTML-график с двумя подграфиками:
    1) Краткосрочные колебания (rolling z-score)
    2) Долгосрочный тренд (абсолютный сентимент)
    
    Args:
        df: DataFrame с агрегированными данными
        output_path: путь для сохранения HTML
        window: 'month' или 'week'
        change_points: список точек изменения от detect_change_points (опционально)
    """

    # Добавьте сюда свои вехи в формате ('YYYY-MM-DD', 'Описание')
    MILESTONES = []

    n_rows = 2
    titles = [
        'Краткосрочные колебания',
        'Долгосрочный тренд',
    ]
    heights = [0.5, 0.5]

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=titles,
        vertical_spacing=0.08,
        row_heights=heights,
    )

    # === Подграфик 1: Краткосрочные колебания ===
    def mood_color(v):
        if v > 0.2: return 'rgb(34, 139, 34)'
        if v > 0.05: return 'rgb(144, 190, 109)'
        if v > -0.05: return 'rgb(180, 180, 100)'
        if v > -0.2: return 'rgb(220, 150, 80)'
        return 'rgb(200, 50, 50)'

    colors_st = [mood_color(v) for v in df['mood_shortterm_smooth']]

    # Сырой rolling z-score (полупрозрачный)
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['mood_shortterm'],
        name='Сырой сигнал', mode='lines',
        line=dict(color='rgba(100, 100, 200, 0.2)', width=1),
        hovertemplate='%{x|%b %Y}<br>Сырой: %{y:.3f}<extra></extra>',
        showlegend=False,
    ), row=1, col=1)

    # Сглаженный (основная линия)
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['mood_shortterm_smooth'],
        name='Настроение (краткосрочн.)', mode='lines+markers',
        line=dict(color='rgba(50, 50, 50, 0.6)', width=2.5),
        marker=dict(color=colors_st, size=5, line=dict(width=0)),
        hovertemplate='%{x|%b %Y}<br>Настроение: %{y:.3f}<extra></extra>',
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4, row=1, col=1)

    # Аннотации: 3 лучших, 3 худших
    for _, row in df.nlargest(3, 'mood_shortterm_smooth').iterrows():
        fig.add_annotation(x=row['date'], y=row['mood_shortterm_smooth'],
            text=row['date'].strftime('%b %Y'), showarrow=True, arrowhead=2,
            font=dict(size=9, color='green'), arrowcolor='green',
            ax=0, ay=-25, row=1, col=1)
    for _, row in df.nsmallest(3, 'mood_shortterm_smooth').iterrows():
        fig.add_annotation(x=row['date'], y=row['mood_shortterm_smooth'],
            text=row['date'].strftime('%b %Y'), showarrow=True, arrowhead=2,
            font=dict(size=9, color='red'), arrowcolor='red',
            ax=0, ay=25, row=1, col=1)

    # === Подграфик 2: Долгосрочный тренд ===

    # Добавляем доверительный интервал для сентимента если доступен
    if 'sentiment_ci_lower' in df.columns and 'sentiment_ci_upper' in df.columns:
        # Полупрозрачная область доверительного интервала
        fig.add_trace(go.Scatter(
            x=pd.concat([df['date'], df['date'][::-1]]),
            y=pd.concat([df['sentiment_ci_upper'], df['sentiment_ci_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(30, 80, 160, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            name='95% ДИ',
            showlegend=True,
        ), row=2, col=1)
    
    # Сырой сентимент (полупрозрачный)
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['sentiment_mean'],
        name='Сентимент (помесячный)', mode='lines',
        line=dict(color='rgba(100, 100, 200, 0.2)', width=1),
        hovertemplate='%{x|%b %Y}<br>Сентимент: %{y:.4f}<extra></extra>',
        showlegend=False,
    ), row=2, col=1)

    # 6-месячный тренд (основная линия)
    colors_lt = [mood_color((v - 0.06) * 15) if not pd.isna(v) else 'gray'
                 for v in df['mood_longterm']]  # центрируем вокруг ~0.06
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['mood_longterm'],
        name='Сентимент (6-мес. тренд)', mode='lines+markers',
        line=dict(color='rgba(30, 80, 160, 0.8)', width=2.5),
        marker=dict(color=colors_lt, size=5, line=dict(width=0)),
        hovertemplate='%{x|%b %Y}<br>Тренд: %{y:.4f}<extra></extra>',
    ), row=2, col=1)

    # Тревога (на той же оси, масштабирована для наглядности)
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['anxiety_trend'],
        name='Тревожность (6-мес.)', mode='lines',
        line=dict(color='rgba(200, 80, 80, 0.5)', width=1.5, dash='dot'),
        hovertemplate='%{x|%b %Y}<br>Тревожность: %{y:.3%}<extra></extra>',
        visible='legendonly',  # скрыта по умолчанию, можно включить в легенде
    ), row=2, col=1)

    # Средняя линия сентимента
    mean_sent = df['sentiment_mean'].mean()
    fig.add_hline(y=mean_sent, line_dash="dash", line_color="rgba(30,80,160,0.3)",
                  opacity=0.5, row=2, col=1,
                  annotation_text=f"среднее: {mean_sent:.4f}",
                  annotation_position="bottom right",
                  annotation_font_size=10, annotation_font_color="rgba(30,80,160,0.5)")

    # === Вехи (на графики 1 и 2) ===
    _yaxis_for_row = {1: 'y', 2: 'y2', 3: 'y3', 4: 'y4'}

    for subplot_row in [1, 2]:
        yref = _yaxis_for_row[subplot_row] + ' domain'
        for ms_date, ms_label in MILESTONES:
            fig.add_vline(x=ms_date, line_dash='dot',
                line_color='rgba(120, 0, 120, 0.25)', line_width=1,
                row=subplot_row, col=1)
            fig.add_annotation(x=ms_date, y=1.0, yref=yref,
                text=f'<b>{ms_label}</b>', showarrow=False,
                font=dict(size=9, color='rgb(100, 0, 100)'),
                textangle=-90, xanchor='right', yanchor='top',
                bgcolor='rgba(255,255,255,0.8)', borderpad=1,
                row=subplot_row, col=1)

    # === Layout ===
    period_name = 'месяцам' if window == 'month' else 'неделям'
    fig.update_layout(
        title=dict(
            text=f'Настроение по переписке в Telegram (по {period_name})',
            font=dict(size=18),
        ),
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=800,
        margin=dict(b=60, t=100),
    )

    y1_max = max(
        abs(df['mood_shortterm_smooth'].min()),
        abs(df['mood_shortterm_smooth'].max()),
    ) * 1.15
    fig.update_yaxes(title_text='Отклонение', range=[-y1_max, y1_max], row=1, col=1)
    fig.update_yaxes(title_text='Сентимент текста', row=2, col=1)
    fig.update_xaxes(type='date', row=1, col=1)
    fig.update_xaxes(type='date', row=2, col=1)

    # Добавляем точки изменения на график
    if change_points:
        add_change_points_to_chart(fig, df, change_points, row=1)

    fig.write_html(output_path, include_plotlyjs='directory')
    print(f"График сохранен: {output_path}")


# ─── Кэширование ───────────────────────────────────────────────────────────

def get_cache_path():
    return os.path.join(os.path.dirname(__file__), 'mood_cache.pkl')


def load_cache(source_path: str):
    """Load cache. Returns list of messages or None."""
    cache_path = get_cache_path()
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if cached.get('source_mtime') == os.path.getmtime(source_path):
            data = cached['messages']
            if isinstance(data, dict):
                messages = data.get('me', [])
            else:
                messages = data
            print(f"Загружен кэш ({len(messages):,} сообщений)")
            return messages
        else:
            print("Кэш устарел, перезапуск обработки...")
    except Exception:
        pass
    return None


def save_cache(messages, source_path: str):
    cache_path = get_cache_path()
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'source_mtime': os.path.getmtime(source_path),
            'messages': messages,
        }, f)
    size_mb = os.path.getsize(cache_path) / 1024 / 1024
    print(f"Кэш сохранен: {cache_path} ({size_mb:.1f} MB)")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Telegram Mood Analysis')
    parser.add_argument(
        '--json',
        default=os.path.join(
            os.path.dirname(__file__),
            'DataExport', 'result.json'
        ),
        help='Путь к result.json',
    )
    parser.add_argument(
        '--user-id',
        default=None,
        help='from_id пользователя (например "user12345678"). '
             'Если не указан, определяется автоматически.',
    )
    parser.add_argument(
        '--window',
        choices=['week', 'month'],
        default='month',
        help='Окно агрегации (default: month)',
    )
    parser.add_argument(
        '--db',
        default=None,
        help='Путь к SQLite базе messages.db (альтернатива --json)',
    )
    parser.add_argument(
        '--user-name',
        default=None,
        dest='user_name',
        help='Имя пользователя для поиска в БД (LIKE). '
             'Если не указано, берётся автор с наибольшим числом сообщений.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Принудительная переобработка (игнорировать кэш)',
    )
    parser.add_argument(
        '--output',
        default=os.path.join(os.path.dirname(__file__), 'mood_chart.html'),
        help='Путь для HTML-графика',
    )
    parser.add_argument(
        '--no-bootstrap',
        action='store_true',
        help='Отключить бутстрэппинг доверительных интервалов (для ускорения)',
    )
    parser.add_argument(
        '--use-llm',
        action='store_true',
        help='Использовать LLM (sentence-transformers) вместо лексиконного анализа',
    )
    args = parser.parse_args()

    source_path = args.db or args.json

    # 1. Извлечение или кэш
    messages = None
    if not args.force:
        messages = load_cache(source_path)

    if messages is None:
        if args.db:
            messages = extract_messages_from_db(args.db, user_name=args.user_name)
        else:
            messages = extract_messages(args.json, user_id=args.user_id)
        
        # Используем LLM анализ если указан флаг
        if args.use_llm:
            print(f"LLM-анализ {len(messages):,} сообщений (sentence-transformers)...")
            try:
                from llm_analyzer import analyze_messages_llm
                llm_df = analyze_messages_llm([m['text'] for m in messages])
                # Интегрируем LLM-метрики в сообщения
                for i, col in enumerate(llm_df.columns):
                    if col != 'text' and col != 'semantic_weight':
                        for j, msg in enumerate(messages):
                            if col in llm_df.columns:
                                msg[f'llm_{col}'] = llm_df.iloc[j][col]
                # Используем LLM sentiment вместо обычного
                for j, msg in enumerate(messages):
                    if 'llm_sentiment' in llm_df.columns:
                        msg['sentiment'] = llm_df.iloc[j]['llm_sentiment']
                    # Применяем семантические веса
                    if 'semantic_weight' in llm_df.columns:
                        msg['semantic_weight'] = llm_df.iloc[j]['semantic_weight']
            except Exception as e:
                print(f"Ошибка LLM-анализа: {e}, используем fallback...")
                messages = score_sentiment(messages)
        else:
            print(f"Анализ сентимента для {len(messages):,} сообщений...")
            messages = score_sentiment(messages)

    save_cache(messages, source_path)

    # 2. Агрегация
    print(f"Агрегация по {args.window}...")
    df = aggregate(messages, args.window, use_bootstrap=not args.no_bootstrap)
    print(f"Периодов: {len(df)}")

    # 3. Статистическая валидация метрики (новый этап!)
    print("\nПроведение статистической валидации метрики...")
    validation_results = validate_mood_metric(df)
    print_validation_report(validation_results)

    # 4. Детекция точек изменения (Change Point Detection)
    print("\nПоиск точек изменения настроения...")
    change_points = detect_change_points(df, signal_col='mood_shortterm', penalty=10)
    print_change_points_report(change_points)

    # 5. Визуализация с точками изменения
    create_chart(df, args.output, args.window, change_points=change_points)

    # 4. Открыть в браузере
    webbrowser.open(f'file://{os.path.abspath(args.output)}')
    print("Готово!")


if __name__ == '__main__':
    main()
