import torch
import torch.nn as nn
import torch.optim as optim

# 1. Defining the "Brain" Structure
# Analogy: Think of this like a factory. 
# Raw data enters, goes through "Processing Rooms" (Layers), and a result comes out.
class SimpleBrain(nn.Module):
    def __init__(self):
        super(SimpleBrain, self).__init__()
        # Linear layer: A math room that does (input * weight + bias)
        self.layer1 = nn.Linear(1, 10) # Takes 1 number, turns it into 10 patterns
        self.layer2 = nn.Linear(10, 1) # Takes those 10 patterns, turns it into 1 final guess

    def forward(self, x):
        # This is the "conveyor belt" moving data through the rooms
        x = torch.relu(self.layer1(x)) # ReLU is a filter that keeps only positive signals
        x = self.layer2(x)
        return x

# 2. Setup the Model, Loss (The Error Meter), and Optimizer (The Adjuster)
model = SimpleBrain()
criterion = nn.MSELoss() # "Mean Squared Error" - measures how far off the guess was
optimizer = optim.SGD(model.parameters(), lr=0.01) # "Stochastic Gradient Descent" - the guy who turns the knobs

# 3. Simple Training Loop
# Analogy: Practicing a song until it sounds right.
print("--- Training the Brain ---")
inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0]]) # Our "Questions"
targets = torch.tensor([[2.0], [4.0], [6.0], [8.0]]) # Our "Correct Answers" (y = 2x)

for epoch in range(100):
    # 1. Forward pass: Make a guess
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # 2. Backward pass: Find out what went wrong and fix it
    optimizer.zero_grad() # Clear previous notes
    loss.backward()      # Calculate the error
    optimizer.step()     # Turn the knobs to improve

    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

# 4. Testing the Brain
print("\n--- Testing the Brain ---")
test_input = torch.tensor([[5.0]])
predicted = model(test_input)
print(f"Question: What is 2 * 5?")
print(f"AI Guess: {predicted.item():.2f} (Target should be 10.00)")
