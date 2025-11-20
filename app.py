# app.py
import os
import asyncio
import sqlite3
import re
import numpy as np
from datetime import datetime
from typing import List, Tuple

from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

# RAG (лёгкий)
try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None

import numpy as np

# -------------------------------
# Настройки
# -------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("❌ BOT_TOKEN не задан")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
router = Router()

embedding_model = None
knowledge_base: List[Tuple[str, np.ndarray]] = []

# -------------------------------
# Вспомогательные функции
# -------------------------------
def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def init_rag():
    global embedding_model, knowledge_base
    print("⚙️ Загрузка модели...")

    if TextEmbedding:
        embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")

    chunks = []
    
    # 🔹 Города — из Excel
    high_cities = [
        "Москва", "Санкт-Петербург", "Краснодар", "Екатеринбург", "Челябинск",
        "Ростов-на-Дону", "Воронеж", "Ставрополь", "Казань", "Самара", "Нижний Новгород"
    ]
    for city in high_cities:
        chunks.append(f"Доставка в {city}: курьерская (с примеркой), экспресс-доставка из магазина, самовывоз и доставка посредством 5Post.")
    
    low_cities = [
        "Пермь", "Новосибирск", "Уфа", "Волгоград", "Саратов", "Тюмень", "Хабаровск",
        "Владивосток", "Иркутск", "Ярославль", "Кемерово", "Барнаул", "Липецк", "Рязань",
        "Курск", "Тула", "Сочи", "Калининград", "Севастополь", "Томск", "Оренбург",
        "Астрахань", "Иваново", "Махачкала", "Чебоксары", "Брянск", "Мурманск",
        "Череповец", "Владикавказ"
    ]
    for city in low_cities:
        chunks.append(f"Доставка в {city}: только посредством 5Post (бесплатно, 2–5 дней, хранение 7 дней).")

    # 🔹 Доставка — данные из PDF (приоритет!)
    chunks += [
        "Курьерская доставка с примеркой: 590 ₽. Срок по Москве — 1–2 дня, по МО — до 3 дней. Примерка до 15 мин. Оплата при получении: наличные/карта.",
        "Курьерская доставка без примерки: 590 ₽. Оплата онлайн. Сроки те же.",
        "Экспресс-доставка из магазина: 590 ₽, до 3 часов, 1 изделие. Доступна в Столешникове, Афимолле, Океании (Москва), Екатеринбурге, Казани, Новосибирске, Сочи, Уфе, Владивостоке.",
        "Самовывоз из магазина: бесплатно, 2–4 дня, хранение 5 дней.",
        "Самовывоз из 5Post: бесплатно, 2–5 дней, хранение 7 дней, лимит 300 000 ₽.",
        "Изменение времени доставки: после передачи курьеру вы получите SMS со ссылкой.",
        "При отказе от заказа стоимость доставки не возвращается. Промокод не восстанавливается.",
    ]

    # 🔹 Возврат
    chunks += [
        "Возврат/обмен — в течение 14 дней при сохранении товарного вида, ярлыков и чека.",
        "Бельё и купальники из магазина — возврат только при браке. Из онлайн — можно вернуть без брака.",
        "Возврат денег — до 10 дней на карту или по реквизитам. Доставка не возвращается.",
        "Бесплатный вызов курьера для возврата — через личный кабинет на сайте.",
    ]

    # 🔹 Прочее
    chunks += [
        "Шопинг-сессия: 2 часа в личной примерочной, персональное сопровождение, игристое. Бронь аннулируется при опоздании >30 мин.",
        "Программа лояльности: 5 баллов за каждые 100 ₽. 1 балл = 1 ₽. Можно оплатить до 30% покупки.",
        "Подарочные сертификаты: от 500 до 500 000 ₽, срок 3 года.",
        "Производство: Россия, Беларусь, Китай, Турция. Страна — на бирке.",
        "На какой рост рассчитана одежда? — 164 см для XXS–XS, 170 см для S–XL. Точные обмеры — на сайте.",
        "Доп. услуги: подшив длины (текстиль, 2 недели), удаление пиллинга (кашемир, 1 год).",
        "Поддержка: 8-800-500-46-11, cs@12storeez.com",
        "Магазины: Москва — Океания, Метрополис, Европейский, Цветной, Атриум, Авиапарк; регионы — Владивосток, Екатеринбург, Казань, Новосибирск, Сочи, Уфа.",
    ]

    if embedding_model:
        print("   → Генерация эмбеддингов...")
        embeddings = list(embedding_model.embed(chunks))
        knowledge_base = [(t, np.array(e)) for t, e in zip(chunks, embeddings)]
    else:
        knowledge_base = [(t, None) for t in chunks]

    print(f"✅ Загружено {len(knowledge_base)} фрагментов.")

def rag_search(query: str, top_k: int = 2) -> List[str]:
    if not knowledge_base:
        return []
    if embedding_model and knowledge_base[0][1] is not None:
        query_emb = np.array(next(embedding_model.embed([query])))
        scored = [(cosine_similarity(query_emb, emb), text) for text, emb in knowledge_base]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:top_k]]
    else:
        # fallback: fuzzy
        import difflib
        texts = [t for t, _ in knowledge_base]
        return difflib.get_close_matches(query.lower(), [t.lower() for t in texts], n=top_k, cutoff=0.3)

# -------------------------------
# FSM: обратная связь
# -------------------------------
class FeedbackState(StatesGroup):
    waiting_for_rating = State()
    waiting_for_comment = State()

def init_feedback_db():
    conn = sqlite3.connect("/tmp/feedback.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            timestamp TEXT,
            rating INTEGER,
            comment TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_feedback(uid, uname, rating, comment):
    conn = sqlite3.connect("/tmp/feedback.db")
    c = conn.cursor()
    c.execute("INSERT INTO feedback VALUES (NULL, ?, ?, ?, ?, ?)",
              (uid, uname or "", datetime.now().isoformat(), rating, comment))
    conn.commit()
    conn.close()

def apology(name: str) -> str:
    return [
        f"Ой, {name}, простите… Похоже, мой ответ сегодня не прошёл финальную примерку 🙈",
        f"Спасибо, что сказали! Я уже бегу к главному редактору стиля — перепишу этот ответ в новой капсуле 🧵",
        f"Аполинария записала ваше замечание в блокнот с золотой застёжкой. В следующий раз — только идеальный крой ответа ✂️",
    ][hash(name) % 3]

# -------------------------------
# Обработчики
# -------------------------------
@router.message(Command("start"))
async def start(message: Message):
    name = message.from_user.first_name or "друг"
    await message.answer(
        f"Здравствуйте, {name}! 👋\n"
        "Я — Аполинария, ваш личный консультант по стилю от *12 STOREEZ* 🌸\n\n"
        "Могу рассказать о доставке, возврате, шопинг-сессии, размерах — просто спросите.\n"
        "Например:\n"
        "— *Доставка в Казань?*\n"
        "— *Как вернуть купальник?*\n"
        "— *Сколько стоит курьер с примеркой?*\n\n"
        "Готова помочь! О чём поговорим?"
    )

@router.message()
async def handle(message: Message, state: FSMContext):
    text = message.text.strip()
    uid = message.from_user.id
    name = message.from_user.first_name or "гость"

    # Оценка
    if re.fullmatch(r"[1-5]", text):
        await state.update_data(rating=int(text))
        await message.answer("Благодарю за оценку! 💌 А теперь — коротко: что можно улучшить?")
        await state.set_state(FeedbackState.waiting_for_comment)
        return

    # Поиск
    results = rag_search(text, top_k=2)
    if not results:
        ans = (
            "К сожалению, я пока не могу ответить точно — но уже спешу уточнить у коллег! 🏃‍♀️\n"
            "Могу предложить:\n"
            "— Позвонить: *8-800-500-46-11*\n"
            "— Написать: *cs@12storeez.com*"
        )
    else:
        ans = "\n\n".join(results[:2])
        if "доставка" in text.lower():
            ans = "📦 " + ans
        elif "возврат" in text.lower():
            ans = "🔄 " + ans
        elif "шопинг" in text.lower():
            ans = "✨ " + ans

    await message.answer(ans, parse_mode="Markdown")

    await asyncio.sleep(1.5)
    await message.answer(
        "Оцените, пожалуйста, насколько мой ответ был вам полезен? 🌟\n"
        "1 — совсем не помог, 5 — идеально!\n(Просто цифру 1–5)"
    )
    await state.set_state(FeedbackState.waiting_for_rating)

@router.message(FeedbackState.waiting_for_rating)
async def rating(message: Message, state: FSMContext):
    if not re.fullmatch(r"[1-5]", message.text):
        await message.answer("Пожалуйста, отправьте цифру от 1 до 5 🌟")
        return
    await state.update_data(rating=int(message.text))
    await message.answer("Спасибо! А теперь — коротко: что можно улучшить?")
    await state.set_state(FeedbackState.waiting_for_comment)

@router.message(FeedbackState.waiting_for_comment)
async def comment(message: Message, state: FSMContext):
    data = await state.get_data()
    rating = data.get("rating", 3)
    comment_text = message.text or ""
    uid = message.from_user.id
    uname = message.from_user.username
    name = message.from_user.first_name or "гость"

    save_feedback(uid, uname, rating, comment_text)

    if rating <= 2:
        await message.answer(apology(name))
    else:
        await message.answer("Спасибо за тёплые слова! 💌 Мне очень приятно 🌸")

    await state.clear()

# -------------------------------
# Health-check для Render
# -------------------------------
from aiohttp import web

async def healthcheck(_):
    return web.json_response({"status": "ok", "bot": "Аполинария online 🌸"})

def create_web_app():
    app = web.Application()
    app.router.add_get("/health", healthcheck)
    return app

# -------------------------------
# Запуск
# -------------------------------
async def main():
    print("🔧 Инициализация...")
    init_feedback_db()
    init_rag()

    print("🤖 Аполинария запускается...")
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == "__main__":
    import threading
    port = int(os.environ.get("PORT", 8000))
    web_app = create_web_app()
    threading.Thread(target=lambda: web.run_app(web_app, host="0.0.0.0", port=port), daemon=True).start()
    asyncio.run(main())
