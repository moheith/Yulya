import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import keyboard
import os

# 1. Concept: Whisper (The Ears)
# We use the "base" model because it's fast and accurate enough for English.
# "audio.wav" is the temporary "tape recording" of your voice.

def record_audio():
    fs = 44100  # Sample rate
    recording = []
    
    # Check if a microphone is actually available
    try:
        default_device = sd.query_devices(kind='input')
        print(f"--- SYSTEM: Using Microphone: {default_device['name']} ---")
    except:
        print("--- ERROR: No microphone found! ---")
        return None

    print("Yulya is listening... (Release 'Right Ctrl' to stop)")

    # Record in slightly larger 1-second chunks for stability
    while keyboard.is_pressed('right ctrl'):
        chunk = sd.rec(int(1.0 * fs), samplerate=fs, channels=1)
        sd.wait()
        recording.append(chunk)

    if len(recording) == 0:
        return None

    import numpy as np
    full_recording = np.concatenate(recording, axis=0)
    
    wav_path = "data/voice_input.wav"
    wav.write(wav_path, fs, full_recording)
    return wav_path

def transcribe_ears(file_path):
    print("Yulya is processing what you said...")
    # Load the Whisper brain
    model = whisper.load_model("base")
    
    # Transcribe the audio
    result = model.transcribe(file_path)
    print(f"You said: {result['text']}")
    return result['text']

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Example Workflow
    # path = record_audio()
    # text = transcribe_ears(path)
    print("Conceptual ear script ready!")
