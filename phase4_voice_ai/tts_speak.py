from kokoro import KPipeline
import sounddevice as sd
import numpy as np
import os

# 1. Concept: Kokoro TTS (The Voice)
# Analogy: This is like a high-speed printer, but for sound. 
# It converts letters into "waves" that your speakers can move to.

def say_text(text):
    print(f"--- Yulya is speaking: '{text}' ---")
    
    # Initialize the voice pipeline
    # 'a' stands for American English, 'f' for Female
    pipeline = KPipeline(lang_code='a') 
    
    # Generate the audio
    # 'af_heart' is our chosen voice for Yulya's personality
    generator = pipeline(
        text, voice='af_heart', 
        speed=1.1, # Slightly faster to match her energetic personality
        split_pattern=r'\n+'
    )

    # Play each piece of the sentence
    for gs, ps, audio in generator:
        # ps = phonemes (how the words are broken down into sounds)
        # audio = the raw sound data
        sd.play(audio, samplerate=24000)
        sd.wait()

if __name__ == "__main__":
    # Example Workflow
    # say_text("Hey dummy! Stop staring at the screen and talk to me properly!")
    print("Conceptual voice script ready!")
