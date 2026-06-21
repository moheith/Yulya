import torch

# 1. Tensors: The Building Blocks
# Analogy: If NumPy is a food processor, Tensors are a "Smart Food Processor."
# They can run on a GPU (the turbocharger) and they remember every math 
# operation performed on them (so they can "learn" backwards).

print("--- 1. Tensors ---")
# Create a simple tensor
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"Tensor from data:\n{x_data}\n")

# 2. GPU Check
# Analogy: Checking if we have a turbocharger available.
# In AI, GPUs (Graphics Cards) make math 100x faster than a normal CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- 2. Hardware Check ---\nUsing device: {device}\n")

# 3. Gradients (The "Memory" for learning)
# This tells PyTorch to track the math so it can adjust the "Weights" later.
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

# This is the "Learning" step where it calculates how to improve
out.backward()
print("--- 3. Gradients (The 'Adjustment' values) ---")
print(x.grad) # This shows how much 'x' needs to change to get a better result
