import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

class FastVision:
    def __init__(self):
        self.model_id = "openai/clip-vit-base-patch32"
        print(f"--- Loading Fast Vision (CLIP): {self.model_id} ---")
        try:
            # Force safetensors to bypass the PyTorch CVE vulnerability
            self.model = CLIPModel.from_pretrained(self.model_id, use_safetensors=True)
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            self.labels = ["a computer screen with code", "a video game", "a youtube video", "an empty desktop", "a chat application"]
        except Exception as e:
            print(f"Vision Load Error: {e}")
            self.model = None

    def scan_screen(self, image_path="data/last_view.png"):
        if not self.model or not os.path.exists(image_path):
            return "[System Context: Vision Offline]"
            
        try:
            image = Image.open(image_path)
            inputs = self.processor(text=self.labels, images=image, return_tensors="pt", padding=True)
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            
            # Find the highest probability label
            best_idx = probs.argmax().item()
            best_label = self.labels[best_idx]
            
            # Translate label to our [EVENT] format
            if best_label == "a computer screen with code":
                return "[System Context: User is looking at code in an editor]"
            elif best_label == "a video game":
                return "[EVENT: Game_Active]"
            elif best_label == "a youtube video":
                return "[EVENT: User_Procrastination | App: YouTube]"
            elif best_label == "a chat application":
                return "[System Context: User is in a chat app]"
            else:
                return "[System Context: User is on the desktop]"
                
        except Exception as e:
            return f"[System Context: Vision Error - {e}]"

if __name__ == "__main__":
    v = FastVision()
    print(v.scan_screen())
