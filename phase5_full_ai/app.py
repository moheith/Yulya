import gradio as gr
import sys
import os
from PIL import Image

# Ensure modules can find each other
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phase5_full_ai.pipeline import YulyaPipeline

# Initialize the pipeline (Loads the Llama 3 brain)
yulya = YulyaPipeline()

def yulya_interface(user_message, history):
    # Gradio strictly requires this exact format now.
    # If history is empty or in the old tuple format, we rebuild it.
    formatted_history = []
    for msg in history:
        if isinstance(msg, dict):
            formatted_history.append(msg)
        elif isinstance(msg, tuple) or isinstance(msg, list):
            # Convert old (user, bot) tuples to new dicts if they exist
            if msg[0]: formatted_history.append({"role": "user", "content": msg[0]})
            if msg[1]: formatted_history.append({"role": "assistant", "content": msg[1]})

    # Run the brain
    ai_response = yulya.run_cycle(user_text=user_message)
    
    # Append the new messages
    formatted_history.append({"role": "user", "content": user_message})
    formatted_history.append({"role": "assistant", "content": ai_response})
    
    return formatted_history, "data/last_view.png"

with gr.Blocks(title="Project Yulya") as demo:
    gr.Markdown("# 🛠️ Project Yulya: Desktop Companion Interface")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Yulya's Conversation", type="messages") # We MUST use 'messages' type
            msg = gr.Textbox(label="Type to Yulya (or hold Right Ctrl for Voice)")
            clear = gr.Button("Clear Chat")
            
        with gr.Column(scale=1):
            gr.Markdown("### Yulya's Current View")
            screen_preview = gr.Image(label="Last Screenshot", value="data/last_view.png")
            gr.Markdown("---")
            gr.Markdown("**Brain:** Fine-tuned Llama 3 (370 steps)")
            gr.Markdown("**Mode:** Safe Mode (RAM Offload)")
            gr.Markdown("**Status:** ONLINE")

    # Connect the UI components
    msg.submit(yulya_interface, [msg, chatbot], [chatbot, screen_preview])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    # Launch the final dashboard
    demo.launch(share=False)
