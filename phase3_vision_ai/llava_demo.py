import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

# 1. Concept: LLaVA (Large Language-and-Vision Assistant)
# Analogy: Think of LLaVA as a person who is looking at a photo while talking 
# on the phone. They can describe exactly what they see in real-time.
# It "glues" a Vision brain (CLIP) to a Language brain (Llama).

def load_vision_brain():
    model_id = "llava-hf/llava-1.5-7b-hf"
    
    print(f"--- Loading Vision Brain: {model_id} ---")
    
    # In a real local setup, we would use:
    # model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    # processor = AutoProcessor.from_pretrained(model_id)

    print("Conceptual Step: LLaVA takes an Image + a Question.")
    print("Example Question: 'What application is open on this desktop?'")
    
    # Example Workflow:
    # image = Image.open("data/ai_input.png")
    # prompt = "USER: <image>\nWhat is the user doing? ASSISTANT:"
    # inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    # generate_ids = model.generate(**inputs, max_new_tokens=100)
    
    print("\nNext: This output is what Yulya uses to generate her [Pose] and dialogue.")

if __name__ == "__main__":
    load_vision_brain()
