import numpy as np

# 1. Creating a simple Array (The "Grid")
# Think of this like a single shelf of books
print("--- 1. Simple Array ---")
numbers = np.array([10, 20, 30, 40, 50])
print(f"Our numbers: {numbers}")
print(f"Shape (how many numbers): {numbers.shape}\n")

# 2. Creating a 2D Array (The "Bookshelf")
# This is like having multiple rows and columns
print("--- 2. 2D Array (Matrix) ---")
grid = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print("Our grid:")
print(grid)
print(f"Shape (Rows, Columns): {grid.shape}\n")

# 3. Why NumPy is "Smart" (Vectorization)
# In normal Python, to add 10 to every number, you'd need a loop.
# In NumPy, you just do it like math!
print("--- 3. Fast Math ---")
bigger_numbers = numbers + 10 
print(f"Original: {numbers}")
print(f"Plus 10:  {bigger_numbers}\n")

# 4. Generating Data
# AI models often start with "random" guesses.
print("--- 4. Randomness ---")
random_data = np.random.rand(3, 3) # A 3x3 grid of random numbers between 0 and 1
print("Random AI Brain patterns:")
print(random_data)
