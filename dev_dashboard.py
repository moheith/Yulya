import customtkinter as ctk
import sys
import os
import json
from PIL import Image
import threading

# Point to our pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from phase5_full_ai.pipeline import YulyaPipeline

# Set Theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class DeveloperDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Project Yulya: Developer Dashboard")
        self.geometry("1200x800")
        
        self.yulya = YulyaPipeline()
        
        self.setup_ui()
        self.update_memory_panel()

    def setup_ui(self):
        # --- LEFT COLUMN: Chat & Input ---
        self.left_frame = ctk.CTkFrame(self, width=400)
        self.left_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        ctk.CTkLabel(self.left_frame, text="Chat History", font=("Arial", 16, "bold")).pack(pady=5)
        
        self.chat_box = ctk.CTkTextbox(self.left_frame, width=380, height=550, state="disabled")
        self.chat_box.pack(padx=10, pady=5)
        
        self.input_entry = ctk.CTkEntry(self.left_frame, width=300, placeholder_text="Type message...")
        self.input_entry.pack(side="left", padx=10, pady=10)
        self.input_entry.bind("<Return>", self.send_message)
        
        self.send_btn = ctk.CTkButton(self.left_frame, text="Send", width=60, command=self.send_message)
        self.send_btn.pack(side="left", pady=10)

        # --- MIDDLE COLUMN: The "Brain" Transparency ---
        self.mid_frame = ctk.CTkFrame(self, width=400)
        self.mid_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(self.mid_frame, text="Internal Pipeline View", font=("Arial", 16, "bold")).pack(pady=5)
        
        # What is sent to Ollama
        ctk.CTkLabel(self.mid_frame, text="Prompt Sent to LLM:", font=("Arial", 12)).pack(anchor="w", padx=10)
        self.prompt_box = ctk.CTkTextbox(self.mid_frame, height=200)
        self.prompt_box.pack(fill="x", padx=10, pady=5)
        
        # Raw JSON from Ollama
        ctk.CTkLabel(self.mid_frame, text="Raw JSON Output:", font=("Arial", 12)).pack(anchor="w", padx=10)
        self.json_box = ctk.CTkTextbox(self.mid_frame, height=150)
        self.json_box.pack(fill="x", padx=10, pady=5)
        
        # Permanent Memory State
        ctk.CTkLabel(self.mid_frame, text="Live Permanent Memory (user_profile.json):", font=("Arial", 12)).pack(anchor="w", padx=10)
        self.memory_box = ctk.CTkTextbox(self.mid_frame, height=150)
        self.memory_box.pack(fill="x", padx=10, pady=5)

        # --- RIGHT COLUMN: Senses (Vision) ---
        self.right_frame = ctk.CTkFrame(self, width=300)
        self.right_frame.pack(side="right", fill="y", padx=10, pady=10)
        
        ctk.CTkLabel(self.right_frame, text="Yulya's Senses", font=("Arial", 16, "bold")).pack(pady=5)
        
        # Image Display
        self.image_label = ctk.CTkLabel(self.right_frame, text="No Screenshot Yet")
        self.image_label.pack(pady=10)
        
        # Vision Text
        ctk.CTkLabel(self.right_frame, text="Vision Analysis Output:", font=("Arial", 12)).pack(anchor="w", padx=10)
        self.vision_box = ctk.CTkTextbox(self.right_frame, height=100)
        self.vision_box.pack(fill="x", padx=10, pady=5)
        self.vision_box.insert("0.0", "[Placeholder: LLaVA integration pending]")

    def log_chat(self, text):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", text + "\n\n")
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def update_internal_views(self, prompt, raw_json, vis_context):
        self.prompt_box.delete("0.0", "end")
        self.prompt_box.insert("0.0", prompt)
        
        self.json_box.delete("0.0", "end")
        self.json_box.insert("0.0", raw_json)
        
        self.vision_box.delete("0.0", "end")
        self.vision_box.insert("0.0", vis_context)
        
        self.update_memory_panel()
        self.update_screenshot()

    def update_memory_panel(self):
        self.memory_box.delete("0.0", "end")
        profile_json = json.dumps(self.yulya.user_profile, indent=4)
        self.memory_box.insert("0.0", profile_json)

    def update_screenshot(self):
        try:
            img = ctk.CTkImage(light_image=Image.open("data/last_view.png"), size=(250, 150))
            self.image_label.configure(image=img, text="")
        except Exception as e:
            self.image_label.configure(text="Screenshot not found.")

    def process_message(self, user_msg):
        # This runs in a thread so the UI doesn't freeze
        self.log_chat(f"You: {user_msg}")
        
        # Call the actual pipeline
        ai_response, raw_json, vis_context = self.yulya.run_cycle(user_text=user_msg)
        
        # We manually build the prompt here just to SHOW it in the UI before Pipeline uses it
        history_str = self.yulya.get_history_string()
        prompt_preview = f"Profile: {json.dumps(self.yulya.user_profile)}\nHistory: {history_str}\nInput: {vis_context}\n{user_msg}"
        
        # Update the UI
        self.log_chat(f"Yulya: {ai_response}")
        self.update_internal_views(prompt_preview, raw_json, vis_context)

    def send_message(self, event=None):
        msg = self.input_entry.get()
        if msg.strip():
            self.input_entry.delete(0, 'end')
            # Run in background to keep UI responsive
            threading.Thread(target=self.process_message, args=(msg,), daemon=True).start()

if __name__ == "__main__":
    if not os.path.exists("data"): os.makedirs("data")
    app = DeveloperDashboard()
    app.mainloop()
