import json

# 1. The Dataset Format
# Analogy: A Flashcard. 
# One side has the "Question" (Instruction), the other has the "Answer" (Output).
# If there is extra context (like an image or a previous sentence), it goes in "Input".

def prepare_data():
    data_path = "data/training_data.json"
    
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"--- Dataset Loaded ---")
    print(f"Total training examples: {len(dataset)}")
    
    # AI models don't just read JSON. They need this data to be "tokenized"
    # (turned into numbers) and formatted into a specific "Prompt Template".
    
    print("\nConceptual Step:")
    print("The model sees: 'Below is an instruction... Respond appropriately.'")
    print(f"Example 1 Instruction: {dataset[0]['instruction']}")
    print(f"Example 1 Expected Output: {dataset[0]['output']}")

if __name__ == "__main__":
    prepare_data()
