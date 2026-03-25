"""Génère le dataset kana à partir des polices système + téléchargées.

Pour chaque police × chaque kana × plusieurs tailles de rendu,
génère des images 28×28 pixels, fond noir, tracé blanc.

96 classes : 48 hiragana (0-47) + 48 katakana (48-95)

Produit :
  - data/kana_dataset.npz   (images uint8 28×28, labels uint8 0-95)
  - data/labels.json         (mapping index → {kana, romaji, type})
  - data/split_indices.npz   (train/val/test 70/15/15, stratifié, seed=42)
"""

import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent / "data"
FONTS_DIR = DATA_DIR / "fonts"
SEED = 42
IMG_SIZE = 28

# ---------------------------------------------------------------------------
# 96 classes : 48 hiragana (0-47) + 48 katakana (48-95)
# ---------------------------------------------------------------------------
HIRAGANA = [
    ("あ", "a"), ("い", "i"), ("う", "u"), ("え", "e"), ("お", "o"),
    ("か", "ka"), ("き", "ki"), ("く", "ku"), ("け", "ke"), ("こ", "ko"),
    ("さ", "sa"), ("し", "shi"), ("す", "su"), ("せ", "se"), ("そ", "so"),
    ("た", "ta"), ("ち", "chi"), ("つ", "tsu"), ("て", "te"), ("と", "to"),
    ("な", "na"), ("に", "ni"), ("ぬ", "nu"), ("ね", "ne"), ("の", "no"),
    ("は", "ha"), ("ひ", "hi"), ("ふ", "fu"), ("へ", "he"), ("ほ", "ho"),
    ("ま", "ma"), ("み", "mi"), ("む", "mu"), ("め", "me"), ("も", "mo"),
    ("や", "ya"), ("ゆ", "yu"), ("よ", "yo"),
    ("ら", "ra"), ("り", "ri"), ("る", "ru"), ("れ", "re"), ("ろ", "ro"),
    ("わ", "wa"), ("ゐ", "wi"), ("ゑ", "we"), ("を", "wo"), ("ん", "n"),
]

KATAKANA = [
    ("ア", "a"), ("イ", "i"), ("ウ", "u"), ("エ", "e"), ("オ", "o"),
    ("カ", "ka"), ("キ", "ki"), ("ク", "ku"), ("ケ", "ke"), ("コ", "ko"),
    ("サ", "sa"), ("シ", "shi"), ("ス", "su"), ("セ", "se"), ("ソ", "so"),
    ("タ", "ta"), ("チ", "chi"), ("ツ", "tsu"), ("テ", "te"), ("ト", "to"),
    ("ナ", "na"), ("ニ", "ni"), ("ヌ", "nu"), ("ネ", "ne"), ("ノ", "no"),
    ("ハ", "ha"), ("ヒ", "hi"), ("フ", "fu"), ("ヘ", "he"), ("ホ", "ho"),
    ("マ", "ma"), ("ミ", "mi"), ("ム", "mu"), ("メ", "me"), ("モ", "mo"),
    ("ヤ", "ya"), ("ユ", "yu"), ("ヨ", "yo"),
    ("ラ", "ra"), ("リ", "ri"), ("ル", "ru"), ("レ", "re"), ("ロ", "ro"),
    ("ワ", "wa"), ("ヰ", "wi"), ("ヱ", "we"), ("ヲ", "wo"), ("ン", "n"),
]

ALL_KANA = HIRAGANA + KATAKANA  # indices 0-95

# Polices système Hiragino (macOS)
SYSTEM_FONTS = [
    "/System/Library/Fonts/ヒラギノ角ゴシック W0.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W1.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W2.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W5.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W7.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W8.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W9.ttc",
    "/System/Library/Fonts/ヒラギノ明朝 ProN.ttc",
    "/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc",
]


# Taille de rendu unique — les variations de scale/marge sont couvertes par l'augmentation on the fly
RENDER_SIZE = 64


def render_kana(font: ImageFont.FreeTypeFont, char: str, render_size: int = 64) -> np.ndarray | None:
    """Rend un kana centré, fond noir (0), tracé blanc (255), 28×28.

    Retourne None si le glyphe n'existe pas dans la police.
    """
    canvas = 96
    img = Image.new("L", (canvas, canvas), 0)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if w < 3 or h < 3:
        return None

    x = (canvas - w) // 2 - bbox[0]
    y = (canvas - h) // 2 - bbox[1]
    draw.text((x, y), char, fill=255, font=font)

    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.array(img)


def collect_fonts() -> list[tuple[str, str]]:
    """Collecte toutes les polices valides (Google Fonts + système).

    Retourne une liste de (nom, chemin) dédupliquée par rendu visuel.
    """
    font_paths = []

    # Google Fonts téléchargées
    for path in sorted(FONTS_DIR.glob("*.ttf")):
        font_paths.append((path.stem, str(path)))

    # Système
    for path in SYSTEM_FONTS:
        if os.path.exists(path):
            name = os.path.basename(path).replace(".ttc", "")
            font_paths.append((name, path))

    # Filtrer : garder celles qui rendent les kana avec un design unique
    valid = []
    seen_hashes = set()

    for name, path in font_paths:
        try:
            font = ImageFont.truetype(path, 64)
            img = render_kana(font, "あ")
            if img is None:
                continue
            h = hashlib.md5(img.tobytes()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            valid.append((name, path))
        except Exception:
            continue

    return valid


def generate_labels() -> None:
    """Génère labels.json."""
    labels = []
    for i, (kana, romaji) in enumerate(HIRAGANA):
        labels.append({"index": i, "kana": kana, "romaji": romaji, "type": "hiragana"})
    offset = len(HIRAGANA)
    for i, (kana, romaji) in enumerate(KATAKANA):
        labels.append({"index": offset + i, "kana": kana, "romaji": romaji, "type": "katakana"})

    with open(DATA_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Collecter les polices
    print("Collecte des polices...")
    fonts = collect_fonts()
    print(f"  {len(fonts)} polices uniques")

    # 2. Générer les images (1 taille par police, l'augmentation couvre le reste)
    print(f"\nGénération du dataset ({len(fonts)} polices × {len(ALL_KANA)} kana)...")
    images = []
    labels = []
    skipped = 0

    for font_name, font_path in fonts:
        try:
            font = ImageFont.truetype(font_path, RENDER_SIZE)
        except Exception:
            continue

        for label_idx, (kana, _) in enumerate(ALL_KANA):
            img = render_kana(font, kana, RENDER_SIZE)
            if img is None:
                skipped += 1
                continue
            images.append(img)
            labels.append(label_idx)

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.uint8)

    print(f"  {len(images)} images générées ({skipped} skipped)")
    print(f"  {len(np.unique(labels))} classes")
    print(f"  Shape: {images.shape}")

    # 3. Sauvegarder
    out_path = DATA_DIR / "kana_dataset.npz"
    np.savez_compressed(out_path, images=images, labels=labels)
    print(f"  → {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB)")

    # 4. Split train/val/test : 70/15/15
    indices = np.arange(len(images))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=SEED, stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=SEED, stratify=labels[temp_idx],
    )

    np.savez(DATA_DIR / "split_indices.npz",
             train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    n_hira = len(HIRAGANA)
    print(f"\nSplit (seed={SEED}):")
    for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        sl = labels[idx]
        print(f"  {name:5s}: {len(idx):4d} ({len(idx)/len(images)*100:.0f}%) "
              f"— {np.sum(sl < n_hira)} hira + {np.sum(sl >= n_hira)} kata")

    # 5. Labels
    generate_labels()
    print(f"\nlabels.json généré ({len(ALL_KANA)} classes)")

    print("\nTerminé.")


if __name__ == "__main__":
    main()
