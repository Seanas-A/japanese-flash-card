"""Application de dessin interactif pour tester le modèle kana CNN.

Dessinez un kana à la souris, les prédictions top-5 s'affichent en temps réel.
"""

import json
import platform
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from model import StochasticDepth  # noqa: F401 — nécessaire pour charger le modèle
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import font_manager as fm

# --- Config ---
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
MODELS_DIR = SCRIPT_DIR / "models"
IMG_SIZE = 28
CANVAS_SIZE = 420
BRUSH_SIZE = 24

# Police japonaise multiplateforme
if platform.system() == "Darwin":
    matplotlib.rcParams["font.family"] = "Hiragino Sans"
else:
    _jp_font = DATA_DIR / "fonts" / "NotoSansJP-Regular.ttf"
    if _jp_font.exists():
        fm.fontManager.addfont(str(_jp_font))
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=str(_jp_font)).get_name()

# Couleurs
BG = "#1a1a2e"
BG_CARD = "#16213e"
ACCENT = "#e94560"
ACCENT2 = "#0f3460"
TEXT = "#eaeaea"
HIRAGANA_COLOR = "#e94560"
KATAKANA_COLOR = "#4ea8de"
KANJI_COLOR = "#2ecc71"


def main():
    model = tf.keras.models.load_model(str(MODELS_DIR / "best_model.keras"))
    with open(DATA_DIR / "labels.json") as f:
        label_info = json.load(f)

    print("Modèle chargé. Dessinez un kana !")

    # --- Fenêtre principale ---
    root = tk.Tk()
    root.title("Kana CNN — Dessin & Prédiction")
    root.configure(bg=BG)
    root.resizable(False, False)

    # Frame principale
    main_frame = tk.Frame(root, bg=BG)
    main_frame.pack(padx=20, pady=20)

    # === Colonne gauche : dessin ===
    left_frame = tk.Frame(main_frame, bg=BG)
    left_frame.pack(side=tk.LEFT, padx=(0, 20))

    tk.Label(
        left_frame, text="Dessinez un kana", fg=TEXT, bg=BG,
        font=("Helvetica", 18, "bold"),
    ).pack(pady=(0, 10))

    # Cadre autour du canvas
    canvas_frame = tk.Frame(left_frame, bg=ACCENT, padx=3, pady=3)
    canvas_frame.pack()

    canvas = tk.Canvas(
        canvas_frame, width=CANVAS_SIZE, height=CANVAS_SIZE,
        bg="black", cursor="crosshair", highlightthickness=0,
    )
    canvas.pack()

    pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    pil_draw = ImageDraw.Draw(pil_image)

    state = {"last_x": None, "last_y": None, "drawing": False}

    def start_draw(event):
        state["drawing"] = True
        state["last_x"] = event.x
        state["last_y"] = event.y

    def draw(event):
        if not state["drawing"]:
            return
        x, y = event.x, event.y
        lx, ly = state["last_x"], state["last_y"]
        canvas.create_line(
            lx, ly, x, y, fill="white", width=BRUSH_SIZE,
            capstyle=tk.ROUND, joinstyle=tk.ROUND,
        )
        pil_draw.line([(lx, ly), (x, y)], fill=255, width=BRUSH_SIZE)
        state["last_x"] = x
        state["last_y"] = y

    def stop_draw(event):
        state["drawing"] = False
        predict()

    canvas.bind("<ButtonPress-1>", start_draw)
    canvas.bind("<B1-Motion>", draw)
    canvas.bind("<ButtonRelease-1>", stop_draw)

    # Boutons
    btn_frame = tk.Frame(left_frame, bg=BG)
    btn_frame.pack(pady=12)

    tk.Button(
        btn_frame, text="Effacer", command=lambda: clear(),
        font=("Helvetica", 13, "bold"), fg="white", bg=ACCENT,
        activebackground="#c0392b", activeforeground="white",
        padx=24, pady=8, relief=tk.FLAT, cursor="hand2",
    ).pack()

    # Indication
    tk.Label(
        left_frame, text="Relâchez le clic pour prédire", fg="#888", bg=BG,
        font=("Helvetica", 10),
    ).pack()

    # === Colonne droite : prédictions ===
    right_frame = tk.Frame(main_frame, bg=BG)
    right_frame.pack(side=tk.LEFT, fill=tk.BOTH)

    tk.Label(
        right_frame, text="Prédictions", fg=TEXT, bg=BG,
        font=("Helvetica", 18, "bold"),
    ).pack(pady=(0, 10))

    fig, (ax_img, ax_bar) = plt.subplots(
        1, 2, figsize=(8, 5), gridspec_kw={"width_ratios": [1, 3]},
        facecolor=BG_CARD,
    )
    fig.subplots_adjust(left=0.05, right=0.88, wspace=0.3, top=0.88, bottom=0.08)

    chart_canvas = FigureCanvasTkAgg(fig, master=right_frame)
    chart_canvas.get_tk_widget().pack()

    # Légende
    legend_frame = tk.Frame(right_frame, bg=BG)
    legend_frame.pack(pady=(8, 0))
    tk.Label(legend_frame, text="\u25cf Hiragana", fg=HIRAGANA_COLOR, bg=BG,
             font=("Helvetica", 11)).pack(side=tk.LEFT, padx=(0, 16))
    tk.Label(legend_frame, text="\u25cf Katakana", fg=KATAKANA_COLOR, bg=BG,
             font=("Helvetica", 11)).pack(side=tk.LEFT, padx=(0, 16))
    tk.Label(legend_frame, text="\u25cf Kanji", fg=KANJI_COLOR, bg=BG,
             font=("Helvetica", 11)).pack(side=tk.LEFT)

    def clear_chart():
        ax_img.clear()
        ax_bar.clear()
        ax_img.set_facecolor("black")
        ax_bar.set_facecolor(BG_CARD)
        ax_img.axis("off")
        ax_bar.axis("off")
        ax_bar.text(
            0.5, 0.5, "Dessinez pour\nvoir les prédictions",
            transform=ax_bar.transAxes, ha="center", va="center",
            fontsize=13, color="#555", style="italic",
        )
        chart_canvas.draw()

    def clear():
        canvas.delete("all")
        pil_draw.rectangle([(0, 0), (CANVAS_SIZE, CANVAS_SIZE)], fill=0)
        clear_chart()

    clear_chart()

    def predict():
        img_28 = pil_image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.array(img_28).astype(np.float32) / 255.0

        if arr.max() < 0.05:
            return

        x_input = arr[np.newaxis, ..., np.newaxis]
        proba = model.predict(x_input, verbose=0)[0]
        top5_idx = np.argsort(proba)[-5:][::-1]

        # Image 28×28
        ax_img.clear()
        ax_img.imshow(arr, cmap="gray", interpolation="nearest")
        ax_img.set_title("28×28", color=TEXT, fontsize=11, pad=8)
        ax_img.axis("off")
        ax_img.set_facecolor("black")

        # Bar chart top-5
        ax_bar.clear()
        kana_labels = []
        for i in top5_idx:
            info = label_info[i]
            char = info.get("char", info.get("kana", "?"))
            reading = info.get("romaji", info.get("on_reading", ""))
            kana_labels.append(f"{char}  {reading}")

        def _color_for(info):
            t = info.get("type", "")
            if t == "hiragana":
                return HIRAGANA_COLOR
            elif t == "katakana":
                return KATAKANA_COLOR
            return KANJI_COLOR

        colors = [_color_for(label_info[i]) for i in top5_idx]
        bars = ax_bar.barh(
            kana_labels[::-1],
            [proba[i] for i in top5_idx[::-1]],
            color=colors[::-1],
            height=0.6,
            edgecolor="none",
        )

        ax_bar.set_xlim(0, 1)
        ax_bar.set_title("Top 5", color=TEXT, fontsize=13, fontweight="bold", pad=10)
        ax_bar.set_facecolor(BG_CARD)
        ax_bar.tick_params(colors=TEXT, labelsize=13)
        ax_bar.spines["bottom"].set_color("#444")
        ax_bar.spines["left"].set_color("#444")
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)

        for bar, idx in zip(bars, top5_idx[::-1]):
            pct = proba[idx]
            ax_bar.text(
                bar.get_width() + 0.03,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1%}",
                va="center", fontsize=12, color=TEXT, fontweight="bold",
            )

        chart_canvas.draw()

    root.mainloop()


if __name__ == "__main__":
    main()
