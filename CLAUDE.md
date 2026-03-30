# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Japanese character recognition CNN (2232 classes: 48 hiragana + 48 katakana + 2136 Jōyō kanji). Dataset generated from 100+ Japanese fonts (auto-discovered from Google Fonts). Exported to CoreML + TF.js. Project language is French.

## Commands

```bash
cd kana-cnn
pip install -r requirements.txt
python prepare_dataset.py             # discover fonts + render all chars → dataset
python prepare_dataset.py --kana-only # kana only (96 classes, faster)
python train.py                       # train CNN
python evaluate.py                    # metrics + confusion matrix
python export.py                      # CoreML + TF.js
```

## Key files

- `prepare_dataset.py` — unified pipeline: auto-discovers Japanese fonts from Google Fonts, downloads them, renders all characters (kana + kanji) as 28x28 images, outputs npz + labels.json + split indices
- `joyo_kanji.py` — extracts 2136 Jōyō kanji from kanjidic2.xml (EDRDG), caches in data/joyo_kanji.json
- `train.py` — ResNet-18 + Attention CNN with on-the-fly augmentation (dynamic NUM_CLASSES from labels.json)
- `evaluate.py` — accuracy, confusion matrix, per-type breakdown
- `export.py` — CoreML + TF.js export

## Constraints

- All seeds fixed (numpy, tensorflow, random). Split: 70/15/15 stratified, seed=42.
- Preprocessing (black bg, white strokes, normalize to [0,1]) must be reproduced identically in the iPadOS app.
- Augmentation on the fly, train only. Val/test never augmented.
