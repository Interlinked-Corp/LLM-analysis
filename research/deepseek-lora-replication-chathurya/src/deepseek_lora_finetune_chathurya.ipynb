!pip install -q transformers datasets peft accelerate
import torch, os, json, shutil
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset as HFDataset
from google.colab import files

print("Starting DeepSeek LoRA Fine-tuning …")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("\n" + "="*50)
print("Please upload your dataset.jsonl file:")
print("="*50)

uploaded = files.upload()
dataset_file = None
for fname in uploaded:
    if fname.endswith(".jsonl"):
        if fname != "dataset.jsonl":
            os.rename(fname, "dataset.jsonl")
        dataset_file = "dataset.jsonl"
        break
if dataset_file is None:
    raise ValueError("No .jsonl file uploaded!")

with open(dataset_file) as f:
    rows = [ln for ln in f if ln.strip()]
print(f"\n✓ Uploaded {dataset_file}  — {len(rows)} examples")
data = []
with open(dataset_file) as f:
    for idx, ln in enumerate(f, 1):
        if ln.strip():
            try:
                data.append(json.loads(ln))
            except json.JSONDecodeError as e:
                print("Skip line", idx, e)
dataset = HFDataset.from_list(data)
print("Columns:", dataset.column_names)

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

special_tokens = {
    "additional_special_tokens": [
        "<disaster_analysis>", "</disaster_analysis>",
        "<event_summary>", "</event_summary>",
        "<detailed_analysis>", "</detailed_analysis>",
        "<predictions>", "</predictions>",
        "<impacts>", "</impacts>",
        "<mitigation_strategies>", "</mitigation_strategies>",
        "<recommendations>", "</recommendations>",
    ]
}
tokenizer.add_special_tokens(special_tokens)

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.resize_token_embeddings(len(tokenizer))
print("Model & tokenizer loaded")

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

system_prompt = (
    "You are a helpful assistant trained to provide detailed, structured disaster analysis reports. "
    "Output exactly these 7 sections without duplication or extra content, using structured text with "
    "angle-bracket headers (<disaster_analysis>, <event_summary>, etc.). Sections: "
    "<disaster_analysis> (13 subsections), <event_summary>, <detailed_analysis>, "
    "<predictions>, <impacts>, <mitigation_strategies>, <recommendations>. "
    "Do not output JSON."
)

def format_batch(ex):
    text = [
        f"{system_prompt}\n\nInstruction: {i}\n\nResponse:\n{o}"
        for i, o in zip(ex["instruction"], ex["output"])
    ]
    tok = tokenizer(
        text, max_length=1024, padding="max_length", truncation=True, return_tensors="pt"
    )
    tok["labels"] = tok["input_ids"].clone()
    return tok

tok_ds = dataset.map(
    format_batch,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenising dataset",
)

training_args = TrainingArguments(
    output_dir="./deepseek-lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    learning_rate=2e-5,
    fp16=torch.cuda.is_available(),
    logging_steps=5,
    save_steps=20,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_ds,
    tokenizer=tokenizer,
)

steps = (
    len(tok_ds) //
    (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    * training_args.num_train_epochs
)
print("Total training steps:", steps)

torch.cuda.empty_cache()
trainer.train()
print("✓ Training complete")

out_dir = "./deepseek-lora-final"
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
print("Saved to", out_dir)

print("\n" + "="*50)
print("Testing model (forced tag prefix)…")
print("="*50)

required_tags = [
    "<disaster_analysis>", "<event_summary>", "<detailed_analysis>",
    "<predictions>", "<impacts>", "<mitigation_strategies>", "<recommendations>",
]

def gen(instr, max_new=1200):
    prompt = f"{system_prompt}\n\nInstruction: {instr}\n\nResponse:\n<disaster_analysis>"
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_new,
            temperature=0.2,
            top_p=0.8,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=False)

tests = [
    "Analyze the potential impact of a Category 5 hurricane hitting Miami, Florida.",
    "Evaluate the risks and provide analysis for a major earthquake in Los Angeles.",
    "Assess the disaster preparedness for flooding in a coastal city.",
]

for n, t in enumerate(tests, 1):
    print(f"\nTest {n}: {t}")
    print("-"*110)
    reply = gen(t)
    print(reply[:2000] + (" …[truncated]" if len(reply) > 2000 else ""))

    found = [tag for tag in required_tags if tag in reply]
    print(f"\n✓ Found {len(found)}/{len(required_tags)} tags → {found}")
