# PROJECT YULYA: UNSLOTH FINE-TUNING SCRIPT
# This script is designed for Google Colab (Free T4 GPU).
# Goal: Train Llama 3 to act as Yulya using "train_on_responses_only".

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset

# 1. Load the Model & Tokenizer
# Analogy: We are bringing the "Actor" (Llama 3) onto the set.
# We load it in 4-bit to save memory (like shrinking a file to fit on a thumb drive).
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit", # Optimized Llama 3
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 2. Add LoRA Adapters (The "Sticky Notes")
# This is where the learning happens. We don't change the whole brain,
# just add a layer of instructions on top.
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank: How much "space" the AI has to learn. 16 is plenty for personality.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# 3. Setup the Chat Template
# We use the standard Llama 3 template so it understands user vs assistant.
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3", 
)

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # We wrap our training data in the Llama 3 chat format
        parts = [
            {"role": "user", "content": f"{instruction} {input}"},
            {"role": "assistant", "content": output},
        ]
        # This converts the dictionary into a raw string for the model
        text = tokenizer.apply_chat_template(parts, tokenize = False, add_generation_prompt = False)
        texts.append(text)
    return { "text" : texts, }

# 4. Load your data\training_data.json
dataset = load_dataset("json", data_files="data/training_data.json", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# 5. The "Secret Sauce": Train on Responses Only
# Analogy: This tells the actor: "Ignore the human's voice. Only practice your 
# own lines so you stay in character."
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # 60 steps is a good start for a small dataset
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_supported_bf16(),
        bf16 = torch.cuda.is_supported_bf16(),
        logging_steps = 1,
        output_dir = "outputs",
    ),
)

# APPLY THE MASKING (CRITICAL STEP)
# This ensures we don't calculate loss on the user prompt.
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# 6. Start Training
print("--- Starting the Yulya Brain Surgery ---")
trainer.train()

# 7. Save the "Sticky Notes" (The LoRA weights)
# This creates the file we will eventually download to your PC.
model.save_pretrained("yulya_lora_model")
tokenizer.save_pretrained("yulya_lora_model")
print("--- Training Complete! Model saved to 'yulya_lora_model' ---")
