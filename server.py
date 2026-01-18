from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"
LORA_PATH = "./devmentor-lora"

print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    trust_remote_code=True,
    quantization_config=quant_config,
    device_map="auto",
)

print("Loading devMentor LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# 🔥 Merge LoRA into the base model so inference always uses fine-tuned weights
try:
    model = model.merge_and_unload()
except Exception as e:
    # If merge isn't supported for some reason, at least ensure eval() is set
    print(f"Warning: could not merge LoRA: {e}")
model.eval()

app = FastAPI(
    title="devMentor API",
    description="devMentor: Fine-tuned DeepSeek-Coder 1.3B as a structured coding tutor",
)

# CORS for your browser UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # fine for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    max_new_tokens: int = 512


def build_prompt(query: str) -> str:
    system_msg = (
        "You are devMentor, a strict but friendly coding tutor. "
        "You MUST answer ONLY in this exact 7-section format and you MUST include all sections:\n"
        "1. Problem Restatement\n"
        "2. Approach / Intuition\n"
        "3. Step-by-step Explanation\n"
        "4. Final Code (with comments)\n"
        "5. Time and Space Complexity\n"
        "6. Edge Cases\n"
        "7. Sample Test Cases\n"
        "Do not add any text before section 1 or after section 7."
    )
    return f"<system>{system_msg}</system>\n<user>{query}</user>\n<assistant>"


@app.get("/")
def root():
    return {"status": "ok", "message": "devMentor API is running"}


@app.post("/chat")
def chat(req: ChatRequest):
    prompt = build_prompt(req.query)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=False,      # deterministic, better structure
            temperature=0.1,
            top_p=0.9,
        )

    generated = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    print("=" * 40)
    print("RAW GENERATED:")
    print(repr(generated))
    print("=" * 40)

    # 1. Keep only everything after the FIRST <assistant>
    if "<assistant>" in generated:
        answer = generated.split("<assistant>", 1)[1]
    else:
        answer = generated

    # 2. Stop at the NEXT turn tag that starts on a new line, like:
    #    "\n<user>" or "\n<system>" or another "\n<assistant>"
    for tag in ["<user>", "<system>", "<assistant>"]:
        marker = "\n" + tag
        idx = answer.find(marker)
        if idx != -1:
            answer = answer[:idx]
            break

    cleaned = answer.strip()

    # Remove any leftover tags like </assistant>, </user>, </system>
    for junk in ["</assistant>", "</user>", "</system>", "<assistant>", "<user>", "<system>"]:
        cleaned = cleaned.replace(junk, "")

    # Strip again after replacements
    cleaned = cleaned.strip()

    print("CLEANED RESPONSE:")
    print(repr(cleaned))
    print("=" * 40)

    return {"response": cleaned}



