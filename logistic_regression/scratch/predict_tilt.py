import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

# Load CSV
df = pd.read_csv("tilt_data.csv")

# Select features (everything except role and tilted)
features = ['loss_streak', 'session_losses', 'session_games', 'session_seconds', 
            'frustration', 'session_deaths', 'last_game_deaths']
X = df[features].values.astype(float)
y = df['tilted'].values.astype(float)

# Normalize features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

# Initialize weights
n_features = X_norm.shape[1]
weights = np.zeros(n_features)
bias = 0

# Hyperparameters
learning_rate = 0.1
epochs = 10000
n = len(y)

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow

def predict_proba(X, weights, bias):
    """Predict probability of being tilted"""
    z = np.dot(X, weights) + bias
    return sigmoid(z)

def binary_cross_entropy(y, y_hat):
    """Binary cross-entropy loss"""
    epsilon = 1e-15  # Prevent log(0)
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def compute_gradients(X, y, y_hat):
    """Compute gradients for weights and bias"""
    dw = (1/n) * np.dot(X.T, (y_hat - y))
    db = (1/n) * np.sum(y_hat - y)
    return dw, db

loss_history = []
accuracy_history = []

print("Training logistic regression model...")
for epoch in range(epochs):
    # Forward pass
    y_hat = predict_proba(X_norm, weights, bias)
    loss = binary_cross_entropy(y, y_hat)
    loss_history.append(loss)
    
    # Calculate accuracy
    predictions = (y_hat >= 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    accuracy_history.append(accuracy)
    
    # Compute gradients
    dw, db = compute_gradients(X_norm, y, y_hat)
    
    # Update weights
    weights -= learning_rate * dw
    bias -= learning_rate * db
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.2%}")

print(f"\nFinal Accuracy: {accuracy_history[-1]:.2%}")
print(f"Final Loss: {loss_history[-1]:.4f}")

# Feature importance (normalized weights)
print("\nFeature Importance:")
for i, feature in enumerate(features):
    print(f"  {feature}: {weights[i]:.4f}")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(loss_history)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Binary Cross-Entropy Loss")
ax1.set_title("Training Loss Over Time")
ax1.grid(True, alpha=0.3)

ax2.plot(accuracy_history)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Training Accuracy Over Time")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Prediction function
def predict_tilt(loss_streak, session_losses, session_games, session_seconds, 
                 frustration, session_deaths, last_game_deaths):
    """Predict probability of being tilted"""
    X_input = np.array([[loss_streak, session_losses, session_games, session_seconds,
                         frustration, session_deaths, last_game_deaths]])
    X_input_norm = (X_input - X_mean) / X_std
    prob = predict_proba(X_input_norm, weights, bias)[0]
    return prob

# ===============================
# UI for prediction
# ===============================

root = tk.Tk()
root.title("Overwatch Tilt Predictor (From Scratch)")
root.geometry("400x500")

# Title
tk.Label(root, text="Tilt Probability Predictor", font=("Arial", 14, "bold")).pack(pady=10)

# Input fields
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

tk.Label(input_frame, text="Loss Streak:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
loss_streak_entry = tk.Entry(input_frame)
loss_streak_entry.grid(row=0, column=1, padx=5, pady=5)
loss_streak_entry.insert(0, "0")

tk.Label(input_frame, text="Session Losses:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
session_losses_entry = tk.Entry(input_frame)
session_losses_entry.grid(row=1, column=1, padx=5, pady=5)
session_losses_entry.insert(0, "0")

tk.Label(input_frame, text="Session Games:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
session_games_entry = tk.Entry(input_frame)
session_games_entry.grid(row=2, column=1, padx=5, pady=5)
session_games_entry.insert(0, "1")

tk.Label(input_frame, text="Session Seconds:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
session_seconds_entry = tk.Entry(input_frame)
session_seconds_entry.grid(row=3, column=1, padx=5, pady=5)
session_seconds_entry.insert(0, "0")

tk.Label(input_frame, text="Frustration (1-5):").grid(row=4, column=0, sticky="w", padx=5, pady=5)
frustration_entry = tk.Entry(input_frame)
frustration_entry.grid(row=4, column=1, padx=5, pady=5)
frustration_entry.insert(0, "3")

tk.Label(input_frame, text="Session Deaths:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
session_deaths_entry = tk.Entry(input_frame)
session_deaths_entry.grid(row=5, column=1, padx=5, pady=5)
session_deaths_entry.insert(0, "0")

tk.Label(input_frame, text="Last Game Deaths:").grid(row=6, column=0, sticky="w", padx=5, pady=5)
last_game_deaths_entry = tk.Entry(input_frame)
last_game_deaths_entry.grid(row=6, column=1, padx=5, pady=5)
last_game_deaths_entry.insert(0, "0")

# Result display
result_frame = tk.Frame(root)
result_frame.pack(pady=10)

result_label = tk.Label(result_frame, text="", font=("Arial", 12))
result_label.pack()

probability_label = tk.Label(result_frame, text="", font=("Arial", 10))
probability_label.pack()

def predict_button_clicked():
    try:
        loss_streak = float(loss_streak_entry.get())
        session_losses = float(session_losses_entry.get())
        session_games = float(session_games_entry.get())
        session_seconds = float(session_seconds_entry.get())
        frustration = float(frustration_entry.get())
        session_deaths = float(session_deaths_entry.get())
        last_game_deaths = float(last_game_deaths_entry.get())
        
        prob = predict_tilt(loss_streak, session_losses, session_games, session_seconds,
                           frustration, session_deaths, last_game_deaths)
        
        prediction = "TILTED" if prob >= 0.5 else "NOT TILTED"
        color = "red" if prob >= 0.5 else "green"
        
        result_label.config(text=f"Prediction: {prediction}", fg=color)
        probability_label.config(text=f"Tilt Probability: {prob:.1%}")
        
    except ValueError:
        result_label.config(text="Please enter valid numbers", fg="orange")
        probability_label.config(text="")

# Predict button
predict_button = tk.Button(root, text="Predict Tilt Status", command=predict_button_clicked, 
                          bg="blue", fg="white", font=("Arial", 10, "bold"))
predict_button.pack(pady=10)

# Instructions
tk.Label(root, text="Enter your session stats to predict tilt probability", 
         font=("Arial", 9), fg="gray").pack(pady=5)

root.mainloop()