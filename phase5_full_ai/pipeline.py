import json
import os
import sys
import ollama

# Ensure modules can find each other
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phase3_vision_ai.screen_vision import capture_screen
from phase3_vision_ai.clip_demo import FastVision

class YulyaPipeline:
    def __init__(self):
        self.history = []
        self.user_profile = self.load_user_profile()
        self.vision = FastVision() # Initialize the eyes
        print("--- Yulya Central Pipeline (Brain API Edition): ONLINE ---")

    def load_user_profile(self):
        profile_path = os.path.join("data", "user_profile.json")
        if not os.path.exists(profile_path):
            default_profile = {"user_name": "User"}
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(default_profile, f, indent=4)
            return default_profile
        with open(profile_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_user_profile(self):
        profile_path = os.path.join("data", "user_profile.json")
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(self.user_profile, f, indent=4)
        print("--- Profile Updated Successfully ---")

    def run_cycle(self, user_text):
        if not user_text.strip():
            return "--- No input detected ---", "{}", "[System Context: Vision Offline]"

        # 2. Capture Vision (Screenshot context)
        capture_screen()
        # Use our FastVision (CLIP) model to analyze the image
        vision_context = self.vision.scan_screen()

        # 3. Build the prompt
        history_str = self.get_history_string()
        prompt = f"Profile: {json.dumps(self.user_profile)}\nHistory: {history_str}\nInput: {vision_context}\n{user_text}"

        # 4. Ask Ollama (Lightning Fast)
        try:
            response = ollama.generate(model='yulya', prompt=prompt)
            raw_ai_output = response['response']
        except Exception as e:
            print(f"Ollama Error: Make sure Ollama is running! {e}")
            return "Error connecting to brain.", "{}", vision_context

        # 5. Parse JSON Response
        try:
            start_idx = raw_ai_output.find('{')
            end_idx = raw_ai_output.rfind('}') + 1
            ai_json = json.loads(raw_ai_output[start_idx:end_idx])
            
            memory_update = ai_json.get("memory_update", {})
            full_response = ai_json.get("response", "")
        except Exception as e:
            print(f"Error parsing AI output: {e}. Output was: {raw_ai_output}")
            memory_update = {}
            full_response = raw_ai_output

        # 6. Handle Memory Updates
        if memory_update:
            self.user_profile.update(memory_update)
            self.save_user_profile()

        # 7. Save to Session History (Restored to 10-message memory)
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": full_response})

        # 8. Act (Animation)
        if "[Pose:" in full_response:
            parts = full_response.split("]", 1)
            pose = parts[0] + "]"
            dialogue = parts[1].strip()
        else:
            pose = "[Pose: Neutral]"
            dialogue = full_response

        print(f"ANIMATION: {pose}")
        
        # We now return dialogue, raw string, and vision context for the dashboard
        return full_response, raw_ai_output, vision_context

    def get_history_string(self):
        # We restored this to remember the last 10 messages!
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.history[-10:]])

if __name__ == "__main__":
    yulya = YulyaPipeline()
    print("\n--- Project Yulya Integration Test ---")
