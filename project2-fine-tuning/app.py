import json
from datasets import Dataset
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

# imports do lora
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch


from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")
login(token)


print("OLHA AQUI BEATRIZ")
print("OLHA AQUI BEATRIZ")
print("OLHA AQUI BEATRIZ")
print("OLHA AQUI BEATRIZ")
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


# mudando o dataset para ser otimizado para hugging face
with open("./datasets/musculacao_dataset.json", "r", encoding="utf-8") as f:
    dados = json.load(f)

dataset = Dataset.from_list(dados)
dataset.save_to_disk("ds_otim_hf")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# resposta
prompt = "Como a musculação ajuda na reabilitação de lesões?"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# quantizacao
bnb_config = {
    "load_in_4bit": True,                       # modelo é carregado usando só 4 bits por peso
    "bnb_4bit_use_double_quant": True,          # quantizacao dupla para melhorar a precisao
    "bnb_4bit_quant_type": "nf4",               # tipo de quantizacao
    "bnb_4bit_compute_dtype": torch.float16,    # tipo de numero usado internamente
}

# iniciando modelo
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", quantization_config=bnb_config
)

model = prepare_model_for_kbit_training(model)

# usando o peft para ajustar os pesos do modelo para o lora
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# tokenizacao
def tokenize(example):
    prompt = f"--- Instrucao:\n{example['instruction']}\n\n--- Resposta:\n{example['output']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

dataset_tokenizado = dataset.map(tokenize)

# ajustando parametros de treinamento
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    save_strategy="epoch",
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# treinando
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_tokenizado,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained("model_ft_lora")
tokenizer.save_pretrained("model_ft_lora")
