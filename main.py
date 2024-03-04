import logging
import fastapi
import uvicorn
import telebot
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer

API_TOKEN = '6913204816:AAGj8236SRELPY-CV_0r6nLSrhxIh98WZhs'

WEBHOOK_HOST = '0be8-109-252-150-238.ngrok-free.app'
WEBHOOK_PORT = 8000  # 443, 80, 88 or 8443 (port need to be 'open')
WEBHOOK_LISTEN = '0.0.0.0'  # In some VPS you may need to put here the IP addr

WEBHOOK_URL_BASE = f"https://{WEBHOOK_HOST}"
WEBHOOK_URL_PATH = f"/{API_TOKEN}/"

logger = telebot.logger
telebot.logger.setLevel(logging.INFO)

bot = telebot.TeleBot(API_TOKEN)
app = fastapi.FastAPI(docs=None, redoc_url=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_response(question, model, tokenizer, device):
    model.eval()  # Переключаем модель в режим оценки
    input_ids = tokenizer.encode(question, return_tensors="pt").to(device)

    # Генерация ответа с заданными параметрами
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=100,  # Максимальная длина сгенерированного ответа
        temperature=0.9,  # Температура генерации для управления случайностью
        top_k=50,  # Лимит для top-k sampling
        top_p=0.95,  # Лимит для nucleus sampling, позволяет генерировать более разнообразный ответ
        num_return_sequences=1,  # Количество возвращаемых последовательностей (ответов)
        pad_token_id=tokenizer.eos_token_id,  # ID токена конца строки для завершения генерации
        no_repeat_ngram_size=2  # Предотвращение повторения n-грамм в ответе
    )

    # Декодирование сгенерированного ответа в строку
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    return generated_text

@app.post(f'/{API_TOKEN}/')
async def process_webhook(update: dict):
    if update:
        update = telebot.types.Update.de_json(update)
        bot.process_new_updates([update])
    else:
        return "No update received."

@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    bot.reply_to(message, "Hi I'm Ragnar, king of all vikings. How can I help you?")

@bot.message_handler(func=lambda message: True, content_types=['text'])
def echo_message(message):
    prediction = generate_response(message.text, model, tokenizer, device)
    bot.reply_to(message, prediction)

# Remove webhook, it fails sometimes the set if there is a previous webhook
bot.remove_webhook()
bot.set_webhook(url=WEBHOOK_URL_BASE + WEBHOOK_URL_PATH)

if __name__ == '__main__':
    uvicorn.run(app, host=WEBHOOK_LISTEN, port=WEBHOOK_PORT)
