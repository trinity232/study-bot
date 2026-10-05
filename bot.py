"""
Учебный помощник по почте.

Ты пишешь письмо с задачей (текст, фото, PDF) со своей бауманской почты
на отдельный ящик бота. Бот проверяет ящик, отправляет задачу в бесплатную
нейросеть YandexGPT и присылает решение ответом на твою почту.
"""

import base64
import io
import email
import html
import imaplib
import json
import logging
import os
import re
import smtplib
import time
import urllib.error
import urllib.request
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import formatdate, make_msgid, parseaddr

from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

# ---------- Настройки (берутся из файла .env) ----------
BOT_EMAIL = os.environ["BOT_EMAIL"]            # ящик бота, например bot@yandex.ru
BOT_PASSWORD = os.environ["BOT_PASSWORD"]      # пароль приложения для этого ящика
ALLOWED_SENDER = os.environ["ALLOWED_SENDER"].strip().lower()  # твой бауманский адрес
SECRET_TAG = os.getenv("SECRET_TAG", "").strip()  # необязательно: слово, которое должно быть в теме
IMAP_HOST = os.getenv("IMAP_HOST", "imap.yandex.ru")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.yandex.ru")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
# Нейросеть: YandexGPT (если задан FOLDER_ID) или любой сервис с OpenAI-совместимым API
FOLDER_ID = os.getenv("FOLDER_ID", "").strip()   # ID каталога в Yandex Cloud
API_KEY = os.environ["API_KEY"]
if FOLDER_ID:
    API_URL = os.getenv("API_URL") or "https://ai.api.cloud.yandex.net/v1/chat/completions"
    _model = os.getenv("MODEL") or "yandexgpt/latest"
    MODEL = _model if _model.startswith("gpt://") else f"gpt://{FOLDER_ID}/{_model}"
    AUTH_HEADER = f"Api-Key {API_KEY}"
else:
    API_URL = os.getenv("API_URL") or "https://api.groq.com/openai/v1/chat/completions"
    MODEL = os.getenv("MODEL") or "openai/gpt-oss-120b"
    AUTH_HEADER = f"Bearer {API_KEY}"
# Умеет ли модель читать картинки. YandexGPT не умеет, поэтому по умолчанию выключено
SUPPORTS_IMAGES = os.getenv("SUPPORTS_IMAGES", "") == "1"
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "30"))
# Сколько секунд один запуск в облаке крутится и проверяет почту
RUN_SECONDS = int(os.getenv("RUN_SECONDS", "270"))

IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024    # лимит API на одну картинку
MAX_PDF_BYTES = 30 * 1024 * 1024

SYSTEM_PROMPT = (
    "Ты помощник студента первого курса МГТУ им. Баумана (программная инженерия). "
    "Тебе присылают учебные задачи по почте. Реши задачу и объясни ход решения "
    "по шагам, чтобы студент понял, как прийти к ответу сам. "
    "Отвечай на русском. Ответ уйдёт обычным текстовым письмом, поэтому не используй "
    "Markdown-разметку (никаких **, ##, таблиц); код оформляй просто отступами. "
    "Никогда не используй LaTeX (никаких $, \\frac, \\sqrt, \\cdot). Формулы пиши обычным "
    "текстом и символами Unicode: x², x³, √x, a·b, a/b, ≤, ≥, ≠, π, ∞, ∫, Σ, →. "
    "Например: (x² + 1)/(2x), √(x + 3), sin²x. "
    "Если в ответе есть программный код, ставь строку '=== КОД ===' перед ним и строку "
    "'=== КОНЕЦ КОДА ===' после него, а сам код пиши как есть, без отступов и без ```. "
    "Если условие неполное или неразборчивое, скажи, чего не хватает."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("study-bot")


def decode_str(value):
    return str(make_header(decode_header(value or "")))


def html_to_text(raw):
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</div>", "\n", raw)
    return html.unescape(re.sub(r"<[^>]+>", " ", raw))


def pdf_to_text(data):
    """Достаёт текст из PDF. Сканы (картинки внутри PDF) так не прочитать — их шли фото."""
    reader = PdfReader(io.BytesIO(data))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return "Текст из PDF-вложения:\n" + (text.strip() or "(в PDF нет текста, похоже на скан)")


def extract(msg):
    """Достаёт из письма текст и вложения (картинки, PDF)."""
    plain, rich, files = [], [], []
    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = part.get_content_type()
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        is_attachment = part.get_content_disposition() == "attachment"
        charset = part.get_content_charset() or "utf-8"

        if ctype in IMAGE_TYPES:
            if len(payload) <= MAX_IMAGE_BYTES:
                files.append((ctype, payload))
            else:
                log.warning("Картинка больше 5 МБ пропущена")
        elif ctype == "application/pdf":
            if len(payload) <= MAX_PDF_BYTES:
                plain.append(pdf_to_text(payload))
        elif ctype == "text/plain" and not is_attachment:
            plain.append(payload.decode(charset, errors="replace"))
        elif ctype == "text/html" and not is_attachment:
            rich.append(html_to_text(payload.decode(charset, errors="replace")))

    text = "\n".join(plain) if plain else "\n".join(rich)
    return text.strip(), files


def solve(subject, text, files):
    content = [{
        "type": "text",
        "text": f"Тема письма: {subject}\n\nТекст письма:\n{text or '(текста нет, смотри вложения)'}",
    }]
    if files and not SUPPORTS_IMAGES:
        content[0]["text"] += ("\n\n(К письму приложены картинки, но эта модель их не видит. "
                               "Если без них условие непонятно, попроси прислать задачу текстом.)")
        files = []
    for ctype, data in files:
        b64 = base64.standard_b64encode(data).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{ctype};base64,{b64}"},
        })

    body = json.dumps({
        "model": MODEL,
        "max_tokens": 4000,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    }).encode()
    headers = {
        "Authorization": AUTH_HEADER,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if FOLDER_ID:
        headers["OpenAI-Project"] = FOLDER_ID
    req = urllib.request.Request(API_URL, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Нейросеть вернула ошибку {e.code}: {detail}") from None

    try:
        data = json.loads(raw)
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (ValueError, KeyError, IndexError, TypeError):
        raise RuntimeError(f"Неожиданный ответ нейросети: {raw[:500]}") from None


LATEX_SIMPLE = {
    r"\cdot": "·", r"\times": "×", r"\le": "≤", r"\leq": "≤", r"\ge": "≥", r"\geq": "≥",
    r"\ne": "≠", r"\neq": "≠", r"\pm": "±", r"\infty": "∞", r"\pi": "π", r"\to": "→",
    r"\Rightarrow": "⇒", r"\approx": "≈", r"\int": "∫", r"\sum": "Σ", r"\alpha": "α",
    r"\beta": "β", r"\bmod": "mod", r"\pmod": "mod", r"\mod": "mod", r"\Delta": "Δ", r"\left": "", r"\right": "", r"\,": " ", r"\;": " ",
}
SUPERSCRIPT = str.maketrans("0123456789+-n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ")


CODE_START, CODE_END = "=== КОД ===", "=== КОНЕЦ КОДА ==="


def clean_answer(text):
    """Чистит формулы во всём ответе, кроме блоков кода — код остаётся как есть."""
    text = re.sub(r"```[a-zA-Z+]*\n(.*?)```", lambda m: f"{CODE_START}\n{m.group(1)}{CODE_END}", text, flags=re.S)
    parts = re.split(r"(=== КОД ===.*?=== КОНЕЦ КОДА ===)", text, flags=re.S)
    out = []
    for p in parts:
        if p.startswith(CODE_START):
            body = p[len(CODE_START):-len(CODE_END)].strip("\n")
            out.append("\n" + "-" * 40 + "\n" + body + "\n" + "-" * 40 + "\n")
        else:
            out.append(clean_math(p))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def clean_math(text):
    """Подчищает LaTeX, если нейросеть всё же его вставила, чтобы письмо читалось."""
    for _ in range(3):  # вложенные \frac
        text = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", text)
        text = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\^\{([0-9+\-n]+)\}", lambda m: m.group(1).translate(SUPERSCRIPT), text)
    text = re.sub(r"\^([0-9n])", lambda m: m.group(1).translate(SUPERSCRIPT), text)
    for k, v in sorted(LATEX_SIMPLE.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(re.escape(k) + r"(?![a-zA-Z])", v, text)
    text = re.sub(r"\\(sin|cos|tan|tg|ctg|ln|log|lim|max|min)\b", r"\1", text)
    text = text.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    text = re.sub(r"^[ \t]*\$\$[ \t]*\n", "", text, flags=re.M)   # строки из одних $$
    text = text.replace("$$", "").replace("$", "")
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)              # **жирный**
    text = re.sub(r"^#{1,6} +", "", text, flags=re.M)             # ## заголовки
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_reply(original, body, as_attachment=False):
    reply = EmailMessage()
    subject = decode_str(original["Subject"])
    reply["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    reply["From"] = BOT_EMAIL
    reply["To"] = ALLOWED_SENDER
    # Без даты и Message-ID Яндекс считает письмо подозрительным и режет как спам
    reply["Date"] = formatdate(localtime=True)
    reply["Message-ID"] = make_msgid(domain=BOT_EMAIL.split("@")[-1])
    if original["Message-ID"]:
        reply["In-Reply-To"] = original["Message-ID"]
        reply["References"] = original["Message-ID"]
    if as_attachment:
        reply.set_content("Решение во вложенном файле reshenie.txt.")
        reply.add_attachment(body.encode("utf-8"), maintype="text", subtype="plain",
                             filename="reshenie.txt")
    else:
        reply.set_content(body)
    return reply


def send_reply(original, body):
    """Отправляет ответ. Если Яндекс режет письмо как спам — шлёт решение файлом,
    а если и это не прошло — кладёт ответ в папку StudyBot ящика бота, чтобы не потерять."""
    last_error = None
    for as_attachment in (False, True):
        reply = build_reply(original, body, as_attachment)
        try:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
                smtp.login(BOT_EMAIL, BOT_PASSWORD)
                smtp.send_message(reply)
            if as_attachment:
                log.info("Обычное письмо не прошло, решение отправлено файлом")
            return
        except smtplib.SMTPDataError as e:
            last_error = e
            log.warning("Яндекс не принял письмо (%s), пробую другой способ", e.smtp_code)

    try:
        with imaplib.IMAP4_SSL(IMAP_HOST) as imap:
            imap.login(BOT_EMAIL, BOT_PASSWORD)
            imap.create("StudyBot")  # если папка уже есть, сервер просто вернёт ошибку — это нормально
            imap.append("StudyBot", None, imaplib.Time2Internaldate(time.time()),
                        build_reply(original, body).as_bytes())
        log.error("Ответ не удалось отправить, он сохранён в папке StudyBot ящика бота")
    except Exception:
        log.exception("Не удалось сохранить ответ в папку StudyBot")
    raise last_error


def check_mailbox():
    with imaplib.IMAP4_SSL(IMAP_HOST) as imap:
        imap.login(BOT_EMAIL, BOT_PASSWORD)
        imap.select("INBOX")
        _, data = imap.search(None, "UNSEEN", "FROM", f'"{ALLOWED_SENDER}"')

        for num in data[0].split():
            _, msg_data = imap.fetch(num, "(RFC822)")  # письмо помечается прочитанным
            msg = email.message_from_bytes(msg_data[0][1])

            sender = parseaddr(msg["From"])[1].lower()
            subject = decode_str(msg["Subject"])
            if sender != ALLOWED_SENDER:
                continue
            if SECRET_TAG and SECRET_TAG.lower() not in subject.lower():
                log.info("Пропущено письмо без секретного слова: %s", subject)
                continue

            log.info("Новая задача: %s", subject)
            try:
                text, files = extract(msg)
                answer = clean_answer(solve(subject, text, files))
            except Exception as e:
                log.exception("Не удалось решить задачу")
                answer = f"Не получилось обработать письмо: {e}"
            send_reply(msg, answer)
            log.info("Ответ отправлен")


def main():
    # В облаке запуск ограничен по времени (RUN_SECONDS), потом GitHub запускает бота заново
    deadline = time.time() + RUN_SECONDS if os.getenv("RUN_ONCE") else None
    log.info("Бот запущен, проверяю %s каждые %s с", BOT_EMAIL, POLL_SECONDS)
    while deadline is None or time.time() < deadline:
        try:
            check_mailbox()
        except Exception:
            log.exception("Ошибка при проверке почты, попробую позже")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
