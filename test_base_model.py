import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Setting up 4-bit quantization...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
    )

    prompt = "Write a Python function to check if a number is prime."
    system_prompt = (
        "You are a helpful coding assistant. "
        "Write clean, correct code followed by a short explanation."
    )
    full_prompt = f"<system>{system_prompt}</system>\n<user>{prompt}</user>\n<assistant>"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.4,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n=== MODEL OUTPUT ===\n")
    print(output_text)

if __name__ == "__main__":
    main()
