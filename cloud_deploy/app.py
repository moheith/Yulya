import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os
import json

# 1. CONFIGURATION
MODEL_REPO = "moheith/Yulya" 
MODEL_FILE = "yulya_final.gguf"

print("--- CLOUD STARTUP: Downloading Brain... ---")
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=".")

print("--- INITIALIZING ENGINE ---")
llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
print("--- ENGINE READY! ---")

def yulya_chat_logic(message, history):
    # history is a list of [user, bot] pairs
    history_str = ""
    for h in history:
        history_str += f"User: {h[0]}\nAssistant: {h[1]}\n"

    # 2. Build the FULL prompt (Showing what we send)
    prompt = f"Profile: {{}}\nHistory: {history_str}\nInput: {message}"
    
    full_request = f"<|start_header_id|>system<|end_header_id|>\n\nYou are Yulya. Respond in JSON.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    # 3. Brain Thinking
    output = llm(full_request, max_tokens=128, stop=["<|eot_id|>", "}\n"], echo=False)
    raw_text = output['choices'][0]['text']
    
    # Ensure it ends with a bracket for parsing
    if not raw_text.strip().endswith('}'):
        raw_text += '}'
        
    try:
        start = raw_text.find('{')
        end = raw_text.rfind('}') + 1
        data = json.loads(raw_text[start:end])
        final_reply = data.get("response", raw_text)
    except:
        final_reply = raw_text

    return final_reply, raw_text, full_request

# 4. THE WEB INTERFACE (Clean & Transparent)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌸 Yulya Cloud Dashboard")
    
    with gr.Row():
        # Left Panel: The Chat
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat History")
            msg = gr.Textbox(label="Message Yulya...", placeholder="Say something nice (or don't)...")
            btn = gr.Button("Send")
            clear = gr.Button("Clear Chat")

        # Right Panel: The "Insides" (What you wanted to see!)
        with gr.Column(scale=1):
            gr.Markdown("### 🧠 Internal Brain View")
            with gr.Accordion("Full Prompt Sent to AI", open=False):
                prompt_view = gr.Textbox(label="Raw Prompt", interactive=False, lines=10)
            
            gr.Markdown("### 📦 Raw JSON Response")
            json_view = gr.Textbox(label="Raw Output", interactive=False)
            
            gr.Markdown("### 👁️ Vision Context")
            vision_view = gr.Textbox(value="[System Context: Active on Windows Desktop]", label="Detected Activity", interactive=False)

    def respond(message, chat_history):
        # Trigger the brain
        bot_message, raw_json, full_prompt = yulya_chat_logic(message, chat_history)
        
        # Update everything
        chat_history.append((message, bot_message))
        return "", chat_history, raw_json, full_prompt

    # Wire up the buttons
    btn.click(respond, [msg, chatbot], [msg, chatbot, json_view, prompt_view])
    msg.submit(respond, [msg, chatbot], [msg, chatbot, json_view, prompt_view])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
