"""
Учебный помощник по почте.

Ты пишешь письмо с задачей (текст, фото, PDF) со своей бауманской почты
на отдельный ящик бота. Бот проверяет ящик, отправляет задачу в бесплатную
нейросеть через GitHub Models и присылает решение ответом на твою почту.
"""

import base64
import io
import email
import html
import imaplib
import logging
import os
import re
import smtplib
import time
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import parseaddr

from dotenv import load_dotenv
from openai import OpenAI
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
MODEL = os.getenv("MODEL", "openai/gpt-4.1")
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
    "Если условие неполное или неразборчивое, скажи, чего не хватает."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("study-bot")
# GitHub Models: бесплатно, ключ — токен GitHub (в Actions выдаётся автоматически)
client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=os.environ["GITHUB_TOKEN"],
)


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
    for ctype, data in files:
        b64 = base64.standard_b64encode(data).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{ctype};base64,{b64}"},
        })

    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=4000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def send_reply(original, body):
    reply = EmailMessage()
    subject = decode_str(original["Subject"])
    reply["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    reply["From"] = BOT_EMAIL
    reply["To"] = ALLOWED_SENDER
    if original["Message-ID"]:
        reply["In-Reply-To"] = original["Message-ID"]
        reply["References"] = original["Message-ID"]
    reply.set_content(body)

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
        smtp.login(BOT_EMAIL, BOT_PASSWORD)
        smtp.send_message(reply)


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
                answer = solve(subject, text, files)
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
