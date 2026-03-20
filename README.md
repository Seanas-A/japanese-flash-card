# Japanese Flash Card — Reconnaissance de kana manuscrits

CNN pour reconnaître **96 kana** (48 hiragana + 48 katakana) manuscrits au stylet. Le modèle est exporté en **CoreML** (iOS/iPadOS) et **TF.js** (browser).

## Dataset

Généré à partir de ~40 polices japonaises (Google Fonts + système macOS) couvrant des styles variés : sans-serif, serif, arrondi, calligraphique, manuscrit enfantin, pinceau, pixel, display, outline, etc.

- 96 classes × ~40 polices = ~3840 images 28×28 niveaux de gris (fond noir, tracé blanc)
- Augmentation on the fly pendant l'entraînement (rotation, shift, zoom, shear)

## Installation

```bash
cd kana-cnn
pip install -r requirements.txt
```

## Utilisation

```bash
cd kana-cnn

# 1. Télécharger les polices
python download_fonts.py

# 2. Générer le dataset
python generate_dataset.py

# 3. Entraîner
python train.py

# 4. Évaluer
python evaluate.py

# 5. Exporter en CoreML + TF.js
python export.py
```

## Architecture du CNN

```
Input(28, 28, 1)
→ Conv2D(32, 3, same) → BatchNorm → ReLU → MaxPool(2) → Dropout(0.25)
→ Conv2D(64, 3, same) → BatchNorm → ReLU → MaxPool(2) → Dropout(0.25)
→ Conv2D(128, 3, same) → BatchNorm → ReLU → GlobalAvgPool
→ Dense(256) → BatchNorm → ReLU → Dropout(0.4)
→ Dense(96, softmax)
```

## Preprocessing

> **Important** : ce pipeline doit être reproduit à l'identique dans l'app iPadOS.

1. Image en niveaux de gris 28×28px
2. Normaliser : `pixel / 255.0`
3. Fond noir, tracé blanc. Si l'entrée est inversée : `1.0 - pixel`
4. Shape : `(1, 28, 28, 1)`

## Entraînement

- Split train/val/test : 70/15/15 stratifié, seed=42
- Augmentation (train uniquement) : rotation ±10°, shift 10%, zoom 10%, shear 5°
- Adam (lr=0.001), batch=256, max 50 epochs
- EarlyStopping (patience=8), ReduceLROnPlateau (patience=4, factor=0.5)
- Seeds fixés (numpy, tensorflow, random)

## Structure

```
kana-cnn/
├── data/
│   ├── fonts/               ← polices japonaises (.ttf)
│   ├── kana_dataset.npz     ← images + labels (généré)
│   ├── split_indices.npz    ← indices train/val/test (généré)
│   └── labels.json          ← mapping 96 classes (généré)
├── models/
│   ├── best_model.keras
│   ├── best_model.mlmodel   ← export CoreML
│   └── tfjs/                ← export TF.js
├── plots/                   ← courbes + matrice de confusion
├── download_fonts.py
├── generate_dataset.py
├── train.py
├── evaluate.py
└── export.py
```

## Licence

MIT
