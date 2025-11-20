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

# RAG: sentence-transformers (лёгкая, CPU-friendly)
from sentence_transformers import SentenceTransformer

# -------------------------------
# Настройки
# -------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("❌ BOT_TOKEN не задан в переменных окружения!")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
router = Router()

# Глобальные переменные для RAG
embedding_model = None
knowledge_base: List[Tuple[str, np.ndarray]] = []


# -------------------------------
# Вспомогательные функции
# -------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def init_rag():
    global embedding_model, knowledge_base
    print("⚙️ Инициализация RAG...")

    print("   → Загрузка модели all-MiniLM-L6-v2 (CPU)...")
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    chunks = []

    # 🔹 Города — из файла "добавь_в_города_..."
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
    delivery_facts = [
        "Курьерская доставка с примеркой: 590 ₽. Срок по Москве (в пределах МКАД): заказ до 18:00 — на след. день, после 18:00 — через день. По МО — до 3 дней. Примерка до 15 мин. Оплата: наличные/карта.",
        "Курьерская доставка без примерки: 590 ₽. Сроки те же. Оплата: онлайн (карта, СБП, Подели, Долями, Сплит).",
        "Экспресс-доставка из магазина: 590 ₽, до 3 часов, 1 изделие. Доступна в Столешникове, Афимолле, Океании (Москва), Екатеринбурге, Казани, Новосибирске, Сочи, Уфе, Владивостоке.",
        "Самовывоз из магазина: бесплатно, 2–4 дня, хранение 5 дней. Оплата: онлайн.",
        "Самовывоз из 5Post: бесплатно, 2–5 дней, хранение 7 дней, лимит 300 000 ₽, вес до 14 кг.",
        "При отказе от заказа стоимость доставки не возвращается. Промокод не восстанавливается.",
        "После передачи заказа курьеру вы получите SMS со ссылкой для изменения даты и времени доставки."
    ]
    chunks.extend(delivery_facts)

    # 🔹 Возврат — из PDF
    return_facts = [
        "Возврат/обмен: 14 дней с момента получения. Нужны чек, ярлыки, товарный вид.",
        "Бельё и купальники из магазина — возврат только при браке. Из онлайн — можно вернуть без брака.",
        "Возврат денег: до 10 дней на карту или по реквизитам. Доставка не возвращается.",
        "Бесплатный вызов курьера для возврата — через личный кабинет на сайте."
    ]
    chunks.extend(return_facts)

    # 🔹 Шопинг-сессия — из PDF
    shopping = [
        "Шопинг-сессия: 2 часа в личной примерочной, персональное сопровождение, игристое. Бронь аннулируется при опоздании >30 мин."
    ]
    chunks.extend(shopping)

    # 🔹 Лояльность — из PDF
    loyalty = [
        "Программа лояльности: 5 баллов за каждые 100 ₽. 1 балл = 1 ₽. Можно оплатить до 30% покупки. Срок действия — 365 дней.",
        "Бонусы: двойные баллы за первую покупку (10%), подарок на день рождения."
    ]
    chunks.extend(loyalty)

    # 🔹 Сертификаты — из PDF
    gift = [
        "Подарочные сертификаты: от 500 до 500 000 ₽, срок 3 года. Доступны онлайн и в магазинах (кроме Цветного)."
    ]
    chunks.extend(gift)

    # 🔹 Услуги — из PDF и Excel
    services = [
        "Подшив длины: для текстильных изделий, срок — 2 недели.",
        "Удаление пиллинга: для кашемира, в течение 1 года после покупки."
    ]
    chunks.extend(services)

    # 🔹 Производство — из PDF
    prod = [
        "Производство: Россия, Беларусь, Китай, Турция. Страна указана на бирке."
    ]
    chunks.extend(prod)

    # 🔹 Адреса — из PDF
    addresses = [
        "Москва: Океания, Метрополис, Европейский, Цветной, Атриум, Авиапарк.",
        "Регионы: Владивосток (Калина Молл), Екатеринбург (Галерея Luxury), Казань (Мега), Новосибирск (Галерея), Сочи (Моремолл), Уфа (Планета)."
    ]
    chunks.extend(addresses)

    # 🔹 Размеры, уход, оплата — из Excel (актуализировано под стиль)
    misc = [
        "На какой рост рассчитана одежда? — 164 см для XXS–XS, 170 см для S–XL. Точные обмеры — на сайте.",
        "Как ухаживать за изделиями? — информация на ярлыке и во вкладке «Уход за изделием» на сайте.",
        "Способы оплаты: онлайн — карты, СБП, рассрочка; при получении — наличные/карта (зависит от доставки).",
        "Контакты: 8-800-500-46-11, cs@12storeez.com"
    ]
    chunks.extend(misc)

    print("   → Генерация эмбеддингов...")
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    knowledge_base = [(text, emb) for text, emb in zip(chunks, embeddings)]

    print(f"✅ Загружено {len(knowledge_base)} фрагментов знаний.")


def rag_search(query: str, top_k: int = 2) -> List[str]:
    if not knowledge_base or embedding_model is None:
        return []

    query_emb = embedding_model.encode([query], convert_to_numpy=True)[0]
    texts, embs = zip(*knowledge_base)
    embs = np.array(embs)

    # Косинусное сходство
    scores = (query_emb @ embs.T) / (np.linalg.norm(query_emb) * np.linalg.norm(embs, axis=1))
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [texts[i] for i in top_indices]


# -------------------------------
# FSM: сбор обратной связи
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


def save_feedback(user_id, username, rating, comment):
    conn = sqlite3.connect("/tmp/feedback.db")
    c = conn.cursor()
    c.execute("INSERT INTO feedback (user_id, username, timestamp, rating, comment) VALUES (?, ?, ?, ?, ?)",
              (user_id, username or "", datetime.now().isoformat(), rating, comment))
    conn.commit()
    conn.close()


def generate_apology(name: str) -> str:
    # 🌸 Аполинария — личность: умная, с лёгкой иронией, метафоры из моды
    lines = [
        f"Ой, {name}, простите… Похоже, мой ответ сегодня не прошёл финальную примерку 🙈",
        f"Спасибо, что сказали! Я уже бегу к главному редактору стиля — перепишу этот ответ в новой капсуле 🧵",
        f"Аполинария записала ваше замечание в блокнот с золотой застёжкой. В следующий раз — только идеальный крой ответа ✂️",
        f"Видимо, я сегодня забыла надеть очки внимательности… Спасибо, что помогли их найти 👓",
    ]
    return lines[hash(name) % len(lines)]


# -------------------------------
# Обработчики
# -------------------------------
@router.message(Command("start"))
async def cmd_start(message: Message):
    name = message.from_user.first_name or "друг"
    await message.answer(
        f"Здравствуйте, {name}! 👋\n"
        "Я — Аполинария, ваш личный консультант по стилю от *12 STOREEZ* 🌸\n\n"
        "Могу рассказать о доставке, возврате, шопинг-сессии, размерах — просто спросите.\n"
        "Например:\n"
        "— *Доставка в Казань?*\n"
        "— *Как вернуть купальник?*\n"
        "— *Сколько стоит курьер с примеркой?*\n"
        "— *Есть ли шопинг-сессия?*\n\n"
        "Готова помочь! О чём поговорим?"
    )


@router.message()
async def handle_message(message: Message, state: FSMContext):
    text = message.text.strip()
    user_id = message.from_user.id
    name = message.from_user.first_name or "гость"

    # Если пользователь вводит оценку (1–5)
    if re.fullmatch(r"[1-5]", text):
        await state.update_data(rating=int(text))
        await message.answer("Благодарю за оценку! 💌\nА теперь — коротко: что можно улучшить?")
        await state.set_state(FeedbackState.waiting_for_comment)
        return

    # RAG-поиск
    results = rag_search(text, top_k=2)

    if not results:
        answer = (
            "К сожалению, я пока не могу ответить точно — но уже спешу уточнить у коллег! 🏃‍♀️\n"
            "Могу предложить:\n"
            "— Позвонить: *8-800-500-46-11*\n"
            "— Написать: *cs@12storeez.com*"
        )
    else:
        answer = "\n\n".join(results[:2])
        # Добавим стиля Аполинарии:
        if "доставка" in text.lower():
            answer = "📦 " + answer
        elif "возврат" in text.lower():
            answer = "🔄 " + answer
        elif "шопинг" in text.lower() or "примерочн" in text.lower():
            answer = "✨ " + answer
        elif "цена" in text.lower() or "стоимость" in text.lower():
            answer = "💰 " + answer

    await message.answer(answer, parse_mode="Markdown")

    # Запрос обратной связи через 1.5 сек
    await asyncio.sleep(1.5)
    await message.answer(
        "Оцените, пожалуйста, насколько мой ответ был вам полезен? 🌟\n"
        "1 — совсем не помог, 5 — идеально!\n(Просто отправьте цифру 1–5)"
    )
    await state.set_state(FeedbackState.waiting_for_rating)


@router.message(FeedbackState.waiting_for_rating)
async def process_rating(message: Message, state: FSMContext):
    text = message.text.strip()
    if not re.fullmatch(r"[1-5]", text):
        await message.answer("Пожалуйста, отправьте цифру от 1 до 5 🌟")
        return
    await state.update_data(rating=int(text))
    await message.answer("Спасибо! А теперь — коротко: что можно улучшить?")
    await state.set_state(FeedbackState.waiting_for_comment)


@router.message(FeedbackState.waiting_for_comment)
async def process_comment(message: Message, state: FSMContext):
    data = await state.get_data()
    rating = data.get("rating", 3)
    comment = message.text or ""
    user_id = message.from_user.id
    username = message.from_user.username
    name = message.from_user.first_name or "гость"

    save_feedback(user_id, username, rating, comment)

    if rating <= 2:
        apology = generate_apology(name)
        await message.answer(apology)
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
    threading.Thread(
        target=lambda: web.run_app(web_app, host="0.0.0.0", port=port),
        daemon=True
    ).start()
    asyncio.run(main())
