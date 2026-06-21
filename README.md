# Project Yulya 🛠️

A 100% locally-hosted, multimodal AI desktop companion featuring a custom-trained personality, RAG-based memory, real-time screen vision, and a voice pipeline.

## Features
*   **Custom Fine-Tuned Persona:** Built on top of Llama 3 (8B) using Unsloth/PEFT to give the AI a distinct "Anime Heroine" personality that uses explicit `[Pose]` tags.
*   **Dual-Memory System:** Features short-term conversation sliding windows and a permanent `user_profile.json` identity card that the AI can dynamically update via JSON parsing.
*   **Local Vision:** Uses `pyautogui` to capture the active desktop screen, allowing the AI to "see" your workspace in real-time.
*   **Voice Integration:** Triggers via a Push-to-Talk keyboard listener, using Whisper for Speech-to-Text (STT) and Kokoro for high-quality Text-to-Speech (TTS).
*   **Custom Dashboard:** A dark-themed desktop GUI built with `CustomTkinter` that decouples the UI thread from the AI inference thread.

## Tech Stack
*   **LLM Engine:** Llama 3 8B (4-bit GGUF Quantization) running via Ollama / llama.cpp
*   **Fine-Tuning:** PyTorch, Transformers, PEFT (LoRA)
*   **Audio:** OpenAI Whisper, Kokoro TTS, sounddevice
*   **Vision:** PyAutoGUI
*   **UI:** CustomTkinter

## Setup
1. Ensure Ollama is installed and running the `yulya` model.
2. Install Python dependencies: `pip install -r requirements.txt`
3. Launch the dashboard: `python dev_dashboard.py`

*Note: The `.gguf` weights and `adapter_model.safetensors` are too large for GitHub and must be downloaded separately.*