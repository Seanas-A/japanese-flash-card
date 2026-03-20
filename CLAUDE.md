# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Kana recognition CNN (96 classes: 48 hiragana + 48 katakana). Dataset generated from Japanese fonts. Exported to CoreML + TF.js. Project language is French.

## Commands

```bash
cd kana-cnn
pip install -r requirements.txt
python download_fonts.py      # download Google Fonts
python generate_dataset.py    # fonts → 28x28 images
python train.py               # train CNN
python evaluate.py            # metrics + confusion matrix
python export.py              # CoreML + TF.js
```

## Key files

- `download_fonts.py` — downloads ~30 Japanese fonts from Google Fonts into `data/fonts/`
- `generate_dataset.py` — renders each kana in each font (black bg, white strokes, 28x28), outputs npz + labels.json + split indices
- `train.py` — CNN with on-the-fly augmentation (96 output classes)
- `evaluate.py` — accuracy, confusion matrix, per-type breakdown
- `export.py` — CoreML + TF.js export

## Constraints

- All seeds fixed (numpy, tensorflow, random). Split: 70/15/15 stratified, seed=42.
- Preprocessing (black bg, white strokes, normalize to [0,1]) must be reproduced identically in the iPadOS app.
- Augmentation on the fly, train only. Val/test never augmented.
