"""Application de dessin interactif pour tester le modèle kana CNN.

Dessinez un kana à la souris, les prédictions top-5 s'affichent en temps réel.
"""

import json
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Patch

# --- Config ---
DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
IMG_SIZE = 28
CANVAS_SIZE = 280
BRUSH_SIZE = 18

# Police japonaise (macOS)
matplotlib.rcParams["font.family"] = "Hiragino Sans"


def main():
    # Charger le modèle et les labels
    model = tf.keras.models.load_model(str(MODELS_DIR / "best_model.keras"))
    with open(DATA_DIR / "labels.json") as f:
        label_info = json.load(f)

    print("Modèle chargé. Dessinez un kana !")

    # --- Tkinter ---
    root = tk.Tk()
    root.title("Kana CNN — Dessin & prédiction")
    root.configure(bg="#2b2b2b")

    # Frame principale
    main_frame = tk.Frame(root, bg="#2b2b2b")
    main_frame.pack(padx=10, pady=10)

    # --- Canvas de dessin ---
    left_frame = tk.Frame(main_frame, bg="#2b2b2b")
    left_frame.pack(side=tk.LEFT, padx=(0, 10))

    tk.Label(left_frame, text="Dessinez un kana", fg="white", bg="#2b2b2b",
             font=("Helvetica", 14)).pack(pady=(0, 5))

    canvas = tk.Canvas(left_frame, width=CANVAS_SIZE, height=CANVAS_SIZE,
                       bg="black", cursor="crosshair", highlightthickness=2,
                       highlightbackground="#666")
    canvas.pack()

    # Image PIL pour capturer le dessin (le canvas tk n'exporte pas facilement)
    pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    pil_draw = ImageDraw.Draw(pil_image)

    # État du dessin
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
        # Dessiner sur le canvas tk
        canvas.create_line(lx, ly, x, y, fill="white", width=BRUSH_SIZE,
                           capstyle=tk.ROUND, joinstyle=tk.ROUND)
        # Dessiner sur l'image PIL
        pil_draw.line([(lx, ly), (x, y)], fill=255, width=BRUSH_SIZE)
        state["last_x"] = x
        state["last_y"] = y

    def stop_draw(event):
        state["drawing"] = False
        predict()

    def clear():
        canvas.delete("all")
        pil_draw.rectangle([(0, 0), (CANVAS_SIZE, CANVAS_SIZE)], fill=0)
        clear_chart()

    canvas.bind("<ButtonPress-1>", start_draw)
    canvas.bind("<B1-Motion>", draw)
    canvas.bind("<ButtonRelease-1>", stop_draw)

    # Bouton effacer
    tk.Button(left_frame, text="Effacer", command=clear,
              font=("Helvetica", 12), padx=16, pady=6).pack(pady=8)

    # --- Zone de prédiction (matplotlib) ---
    right_frame = tk.Frame(main_frame, bg="#2b2b2b")
    right_frame.pack(side=tk.LEFT)

    fig, (ax_img, ax_bar) = plt.subplots(
        1, 2, figsize=(7, 3.5), gridspec_kw={"width_ratios": [1, 3]},
        facecolor="#2b2b2b"
    )
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.3)

    chart_canvas = FigureCanvasTkAgg(fig, master=right_frame)
    chart_canvas.get_tk_widget().pack()

    def clear_chart():
        ax_img.clear()
        ax_bar.clear()
        ax_img.set_facecolor("black")
        ax_bar.set_facecolor("#2b2b2b")
        ax_img.axis("off")
        ax_bar.axis("off")
        chart_canvas.draw()

    clear_chart()

    def predict():
        # Resize 280→28, normaliser
        img_28 = pil_image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.array(img_28).astype(np.float32) / 255.0

        # Vérifier qu'il y a bien quelque chose de dessiné
        if arr.max() < 0.05:
            return

        x_input = arr[np.newaxis, ..., np.newaxis]
        proba = model.predict(x_input, verbose=0)[0]
        top5_idx = np.argsort(proba)[-5:][::-1]

        # Image 28×28
        ax_img.clear()
        ax_img.imshow(arr, cmap="gray")
        ax_img.set_title("28×28", color="white", fontsize=10)
        ax_img.axis("off")
        ax_img.set_facecolor("black")

        # Bar chart top-5
        ax_bar.clear()
        kana_labels = [
            f"{label_info[i]['kana']}  {label_info[i]['romaji']}"
            for i in top5_idx
        ]
        colors = [
            "#e74c3c" if label_info[i]["type"] == "hiragana" else "#3498db"
            for i in top5_idx
        ]
        bars = ax_bar.barh(
            kana_labels[::-1],
            [proba[i] for i in top5_idx[::-1]],
            color=colors[::-1],
        )
        ax_bar.set_xlim(0, 1)
        ax_bar.set_title("Top 5", color="white", fontsize=11)
        ax_bar.set_facecolor("#2b2b2b")
        ax_bar.tick_params(colors="white")
        ax_bar.spines["bottom"].set_color("#666")
        ax_bar.spines["left"].set_color("#666")
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)

        for bar, idx in zip(bars, top5_idx[::-1]):
            ax_bar.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{proba[idx]:.1%}",
                va="center", fontsize=10, color="white",
            )

        chart_canvas.draw()

    root.mainloop()


if __name__ == "__main__":
    main()
