import tkinter as tk
from tkinter import ttk, messagebox
import csv
import os
import time

FILE_NAME = "tilt_data.csv"
FIELDS = [
    "loss_streak",
    "session_losses",
    "session_games",
    "session_seconds",
    "frustration",
    "session_deaths",
    "last_game_deaths",
    "role",
    "tilted"
]

# CSV
def ensure_csv_exists():
    if not os.path.exists(FILE_NAME):
        with open(FILE_NAME, "w", newline="") as f:
            csv.writer(f).writerow(FIELDS)

# App
class TiltTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Overwatch Tilt Tracker")
        self.root.geometry("550x900")

        ensure_csv_exists()
        self.reset_session()

        self.main = ttk.Frame(root, padding=20)
        self.main.pack(fill="both", expand=True)

        self.build_idle_ui()

    # Session State
    def reset_session(self):
        self.session_active = False
        self.start_time = None
        self.games = []  # list of dicts: each game
        self.loss_streak = 0

    # UI Builders
    def clear_ui(self):
        for widget in self.main.winfo_children():
            widget.destroy()

    def build_idle_ui(self):
        self.clear_ui()
        ttk.Label(
            self.main,
            text="Overwatch Competitive Tilt Tracker",
            font=("Segoe UI", 16, "bold")
        ).pack(pady=20)

        ttk.Button(
            self.main,
            text="Log New Session",
            command=self.start_session
        ).pack(pady=40)

    def build_session_ui(self):
        self.clear_ui()

        # Timer
        self.timer_label = ttk.Label(
            self.main,
            text="Session Time: 0 sec",
            font=("Segoe UI", 12, "bold")
        )
        self.timer_label.pack(pady=10)

        # Game Loggin Section
        self.game_frame = ttk.LabelFrame(self.main, text="Game Logging", padding=10)
        self.game_frame.pack(fill="x", pady=5)

        # Game Result inline radio buttons
        ttk.Label(self.game_frame, text="Game Result").pack(anchor="w")
        result_frame = ttk.Frame(self.game_frame)
        result_frame.pack(anchor="w", pady=(0,5))
        self.game_result = tk.StringVar(value="win")
        ttk.Radiobutton(result_frame, text="Win", variable=self.game_result, value="win").pack(side="left", padx=5)
        ttk.Radiobutton(result_frame, text="Loss", variable=self.game_result, value="loss").pack(side="left", padx=5)

        ttk.Label(self.game_frame, text="Deaths").pack(anchor="w", pady=(10, 0))
        self.deaths_var = tk.StringVar()
        ttk.Entry(self.game_frame, textvariable=self.deaths_var).pack(fill="x")

        ttk.Label(self.game_frame, text="Frustration (1–5)").pack(anchor="w", pady=(10, 0))
        self.frustration_var = tk.StringVar(value="3")
        ttk.Entry(self.game_frame, textvariable=self.frustration_var).pack(fill="x")

        ttk.Label(self.game_frame, text="Role").pack(anchor="w", pady=(10, 0))
        self.role_var = tk.StringVar(value="dps")
        ttk.Combobox(
            self.game_frame,
            textvariable=self.role_var,
            values=["tank", "dps", "support"],
            state="readonly"
        ).pack(fill="x")

        ttk.Button(self.game_frame, text="Log Game", command=self.log_game).pack(pady=5)
        ttk.Button(self.game_frame, text="Update Selected Game", command=self.update_game).pack(pady=5)

        ttk.Label(self.game_frame, text="Logged Games:").pack(anchor="w", pady=(10, 0))
        self.games_listbox = tk.Listbox(self.game_frame, height=8)
        self.games_listbox.pack(fill="x")
        self.games_listbox.bind("<<ListboxSelect>>", self.on_select_game)

        # Session Stats Section
        self.stats_frame = ttk.LabelFrame(self.main, text="Session Stats", padding=10)
        self.stats_frame.pack(fill="x", pady=5)
        self.summary_label = ttk.Label(self.stats_frame, text="", justify="left")
        self.summary_label.pack(anchor="w")

        # Supervised Learning Section
        self.supervised_frame = ttk.LabelFrame(self.main, text="Supervised Learning", padding=10)
        self.supervised_frame.pack(fill="x", pady=5)
        ttk.Label(self.supervised_frame, text="Tilted overall this session?").pack(anchor="w")

        tilt_frame = ttk.Frame(self.supervised_frame)
        tilt_frame.pack(anchor="w", pady=(0,5))
        self.tilted_var = tk.IntVar(value=0)
        ttk.Radiobutton(tilt_frame, text="No", variable=self.tilted_var, value=0).pack(side="left", padx=5)
        ttk.Radiobutton(tilt_frame, text="Yes", variable=self.tilted_var, value=1).pack(side="left", padx=5)

        ttk.Button(self.main, text="End Session", command=self.end_session).pack(pady=10)

        # Start timer and summary update
        self.update_timer()
        self.update_summary()

    # Logic
    def start_session(self):
        self.reset_session()
        self.session_active = True
        self.start_time = time.time()
        self.build_session_ui()

    def update_timer(self):
        if not self.session_active:
            return
        seconds = int(time.time() - self.start_time)
        self.timer_label.config(text=f"Session Time: {seconds} sec")
        self.root.after(1000, self.update_timer)

    def update_summary(self):
        session_losses = sum(1 for g in self.games if g["result"] == "loss")
        session_deaths = sum(g["deaths"] for g in self.games)
        games_played = len(self.games)
        streak = 0
        for g in reversed(self.games):
            if g["result"] == "loss":
                streak += 1
            else:
                break
        self.loss_streak = streak

        last_game_role = self.games[-1]["role"] if self.games else "N/A"
        last_game_deaths = self.games[-1]["deaths"] if self.games else 0

        # Average session frustration
        avg_frustration = round(sum(g["frustration"] for g in self.games)/len(self.games), 2) if self.games else 0

        summary_text = (
            f"Games Played: {games_played}\n"
            f"Session Losses: {session_losses}\n"
            f"Session Deaths: {session_deaths}\n"
            f"Loss Streak: {self.loss_streak}\n"
            f"Last Game Deaths: {last_game_deaths}\n"
            f"Average Frustration: {avg_frustration}\n"
            f"Last Game Role: {last_game_role}\n"
            f"Session Seconds: {int(time.time() - self.start_time) if self.session_active else 0}"
        )
        self.summary_label.config(text=summary_text)
        self.refresh_listbox()

    def refresh_listbox(self):
        self.games_listbox.delete(0, tk.END)
        for idx, g in enumerate(self.games):
            self.games_listbox.insert(
                tk.END,
                f"{idx+1}: {g['result'].capitalize()} | Deaths: {g['deaths']} | Frust: {g['frustration']} | Role: {g['role']}"
            )

    def log_game(self):
        # Validate deaths
        try:
            deaths = int(self.deaths_var.get())
        except ValueError:
            messagebox.showerror("Error", "Deaths must be a number.")
            return

        # Validate frustration
        try:
            frustration = int(self.frustration_var.get())
            if not 1 <= frustration <= 5:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Frustration must be an integer between 1 and 5.")
            return

        game = {
            "result": self.game_result.get(),
            "deaths": deaths,
            "frustration": frustration,
            "role": self.role_var.get()
        }

        self.games.append(game)
        self.deaths_var.set("")
        self.frustration_var.set("3")
        self.update_summary()

    def on_select_game(self, event):
        if not self.games_listbox.curselection():
            return
        idx = self.games_listbox.curselection()[0]
        g = self.games[idx]
        self.game_result.set(g["result"])
        self.deaths_var.set(str(g["deaths"]))
        self.frustration_var.set(str(g["frustration"]))
        self.role_var.set(g["role"])

    def update_game(self):
        if not self.games_listbox.curselection():
            messagebox.showwarning("Select Game", "Please select a game to update.")
            return
        idx = self.games_listbox.curselection()[0]

        # Validate deaths
        try:
            deaths = int(self.deaths_var.get())
        except ValueError:
            messagebox.showerror("Error", "Deaths must be a number.")
            return

        # Validate frustration
        try:
            frustration = int(self.frustration_var.get())
            if not 1 <= frustration <= 5:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Frustration must be an integer between 1 and 5.")
            return

        self.games[idx] = {
            "result": self.game_result.get(),
            "deaths": deaths,
            "frustration": frustration,
            "role": self.role_var.get()
        }
        self.update_summary()

    def end_session(self):
        if len(self.games) == 0:
            messagebox.showwarning("No Games", "You must log at least one game.")
            return

        session_seconds = int(time.time() - self.start_time)
        session_losses = sum(1 for g in self.games if g["result"] == "loss")
        session_deaths = sum(g["deaths"] for g in self.games)
        games_played = len(self.games)
        last_game = self.games[-1]
        avg_frustration = round(sum(g["frustration"] for g in self.games)/len(self.games), 2)

        row = [
            self.loss_streak,
            session_losses,
            games_played,
            session_seconds,
            avg_frustration,
            session_deaths,
            last_game["deaths"],
            last_game["role"],
            self.tilted_var.get()
        ]

        with open(FILE_NAME, "a", newline="") as f:
            csv.writer(f).writerow(row)

        messagebox.showinfo("Saved", "Session saved successfully!")
        self.session_active = False
        self.build_idle_ui()

# Run 
if __name__ == "__main__":
    root = tk.Tk()
    app = TiltTracker(root)
    root.mainloop()
