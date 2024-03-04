from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

df = pd.read_csv('ragnar_3_csv.csv')

class CharacterDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        message = str(self.data.iloc[index, 0])
        response = str(self.data.iloc[index, 1])
        # Объедини контекста и ответ для обучения в стиле "Вопрос-Ответ"
        input_text = message + " " + response
        inputs = self.tokenizer.encode_plus(
            input_text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long)
        }

# Инициализация токенизатора и модели
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Подготовка датасета и даталоадера
dataset = CharacterDataset(df, tokenizer, max_length=512)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Подготовка к обучению
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(loader))

# Обучение
model.train()
for epoch in range(3):  # 3 эпохи
    for _,batch in enumerate(loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

def generate_response(question, model, tokenizer, device):
    model.eval()  # Переключаем модель в режим оценки
    input_ids = tokenizer.encode(question, return_tensors="pt").to(device)

    # Генерация ответа с заданными параметрами
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=50,  # Максимальная длина сгенерированного ответа
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

# Пример использования функции для генерации ответа
question = "How are you?"  # Вопрос к Рагнару
response = generate_response(question, model, tokenizer, device)
print(f"Generated response: {response}")


