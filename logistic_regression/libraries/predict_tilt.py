import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import tkinter as tk

# Load CSV
df = pd.read_csv("tilt_data.csv")

# Select features (everything except role and tilted)
features = ['loss_streak', 'session_losses', 'session_games', 'session_seconds', 
            'frustration', 'session_deaths', 'last_game_deaths']
X = df[features].values
y = df['tilted'].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Make predictions
y_pred = model.predict(X_scaled)
y_pred_proba = model.predict_proba(X_scaled)[:, 1]

# Evaluate model
accuracy = accuracy_score(y, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=['Not Tilted', 'Tilted']))

# Feature importance
print("\nFeature Importance (Coefficients):")
for i, feature in enumerate(features):
    print(f"  {feature}: {model.coef_[0][i]:.4f}")
print(f"  Intercept: {model.intercept_[0]:.4f}")

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Tilted', 'Tilted'],
            yticklabels=['Not Tilted', 'Tilted'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# Feature importance visualization
plt.figure(figsize=(10, 6))
coefficients = model.coef_[0]
feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients
}).sort_values('Coefficient', key=abs, ascending=False)

colors = ['red' if x > 0 else 'blue' for x in feature_importance['Coefficient']]
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors)
plt.xlabel('Coefficient Value')
plt.title('Feature Importance (Positive = Increases Tilt Probability)')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()

# Prediction distribution
plt.figure(figsize=(10, 6))
plt.hist(y_pred_proba[y == 0], bins=20, alpha=0.5, label='Not Tilted (Actual)', color='green')
plt.hist(y_pred_proba[y == 1], bins=20, alpha=0.5, label='Tilted (Actual)', color='red')
plt.xlabel('Predicted Tilt Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities')
plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
plt.legend()
plt.tight_layout()
plt.show()

# Prediction function
def predict_tilt(loss_streak, session_losses, session_games, session_seconds,
                 frustration, session_deaths, last_game_deaths):
    """Predict probability of being tilted using trained model"""
    X_input = np.array([[loss_streak, session_losses, session_games, session_seconds,
                         frustration, session_deaths, last_game_deaths]])
    X_input_scaled = scaler.transform(X_input)
    prob = model.predict_proba(X_input_scaled)[0, 1]
    return prob

# ===============================
# UI for prediction
# ===============================

root = tk.Tk()
root.title("Overwatch Tilt Predictor (Scikit-learn)")
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