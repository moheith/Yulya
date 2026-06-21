import pyautogui
from PIL import Image
import os

# 1. Taking a Screenshot
# Analogy: This is Yulya "opening her eyes" and blinking.
def capture_screen():
    print("--- Yulya is looking at your screen... ---")
    
    # Take a screenshot of the entire primary monitor
    screenshot = pyautogui.screenshot()
    
    # Save it temporarily so the AI can read it
    save_path = "data/last_view.png"
    screenshot.save(save_path)
    print(f"Screenshot saved to {save_path}")
    
    # 2. Resizing for the AI
    # Analogy: AI models aren't good at looking at giant 4K posters. 
    # They like "postcards." We shrink the image so the AI brain doesn't get overwhelmed.
    with Image.open(save_path) as img:
        img_resized = img.resize((336, 336)) # Standard size for LLaVA
        img_resized.save("data/ai_input.png")
        print("Image optimized for Yulya's vision model.")

if __name__ == "__main__":
    # Create the data folder if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
    capture_screen()
