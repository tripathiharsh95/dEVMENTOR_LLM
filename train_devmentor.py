import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"
DATA_PATH = "data/devmentor_dataset.jsonl"
OUTPUT_DIR = "./devmentor-lora"

def load_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return tokenizer, model

def format_example(example):
    instruction = example["instruction"]
    inp = example.get("input") or ""
    output = example["output"]

    system_msg = (
        "You are devMentor, a friendly coding tutor. "
        "Always answer in this 7-section format:\n"
        "1. Problem Restatement\n"
        "2. Approach / Intuition\n"
        "3. Step-by-step Explanation\n"
        "4. Final Code (with comments)\n"
        "5. Time and Space Complexity\n"
        "6. Edge Cases\n"
        "7. Sample Test Cases\n"
    )

    if inp.strip():
        prompt = (
            f"<system>{system_msg}</system>\n"
            f"<user>{instruction}\n\nAdditional info:\n{inp}</user>\n"
            f"<assistant>{output}</assistant>"
        )
    else:
        prompt = (
            f"<system>{system_msg}</system>\n"
            f"<user>{instruction}</user>\n"
            f"<assistant>{output}</assistant>"
        )

    return {"text": prompt}

def main():
    tokenizer, model = load_tokenizer_and_model()

    dataset = load_dataset("json", data_files=DATA_PATH)["train"]
    dataset = dataset.map(format_example)

    def tokenize(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=1024,
            padding="max_length",
        )
        # For causal LM, we normally predict the next token of the same sequence.
        # So labels are just a copy of input_ids.
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
