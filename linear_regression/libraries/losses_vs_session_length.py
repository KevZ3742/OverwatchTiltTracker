import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import tkinter as tk

# Load CSV
df = pd.read_csv("tilt_data.csv")

x = df['session_seconds'].values.reshape(-1, 1)
y = df['session_losses'].values

# Normalize feature
scaler = StandardScaler()
x_norm = scaler.fit_transform(x)

# Create and train model
model = LinearRegression()
model.fit(x_norm, y)

# Get slope and intercept in original scale
m_norm = model.coef_[0]  # slope for normalized x
b_norm = model.intercept_  # intercept in original scale

# Convert slope back to original scale
m_orig = m_norm / scaler.scale_[0]
b_orig = b_norm - m_norm * scaler.mean_[0] / scaler.scale_[0]

print(f"Slope (m): {m_orig:.4f} → extra losses per second")
print(f"Intercept (b): {b_orig:.4f} → predicted losses at 0 seconds")

# Plot regression line
plt.scatter(df['session_seconds'], y, label="Actual Data")
plt.plot(df['session_seconds'], model.predict(x_norm), color="red", label="Fitted Line")
plt.xlabel("Session Seconds")
plt.ylabel("Session Losses")
plt.title("Session Losses vs Session Length (scikit-learn)")
plt.legend()
plt.show()

# ===============================
# UI to take input and show predicted losses
# ===============================

def predict_session_losses_lib(session_seconds):
    # Normalize the input like training
    session_norm = scaler.transform([[session_seconds]])
    return model.predict(session_norm)[0]

# Create window
root = tk.Tk()
root.title("Overwatch Tilt Predictor (Library)")
root.geometry("300x150")

# Input label and field
tk.Label(root, text="Enter session length (seconds):").pack(pady=5)
session_entry = tk.Entry(root)
session_entry.pack(pady=5)

# Output label
result_label = tk.Label(root, text="Predicted losses will appear here")
result_label.pack(pady=10)

# Function to predict
def predict_button_clicked():
    try:
        session_seconds = float(session_entry.get())
        predicted = predict_session_losses_lib(session_seconds)
        result_label.config(text=f"Predicted losses: {predicted}")
    except ValueError:
        result_label.config(text="Please enter a valid number")

# Button
predict_button = tk.Button(root, text="Predict Losses", command=predict_button_clicked)
predict_button.pack(pady=5)

# Start the UI loop
root.mainloop()
