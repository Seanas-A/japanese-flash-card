"""Télécharge des polices japonaises gratuites depuis Google Fonts pour enrichir le dataset.

Styles variés : serif, sans-serif, arrondi, calligraphique, manuscrit, pinceau, pixel, gras, fin, etc.
Toutes les polices sont PLEINES (pas de contour/outline).
"""

from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

FONTS_DIR = Path(__file__).parent / "data" / "fonts"

# Polices Google Fonts avec kana — sélectionnées pour la variété de style
# Toutes pleines (pas d'outline/creux)
GOOGLE_FONTS = {
    # ===== SANS-SERIF / GOTHIC =====
    "NotoSansJP-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf",
    "MPLUS1p-Thin.ttf": "https://github.com/google/fonts/raw/main/ofl/mplus1p/MPLUS1p-Thin.ttf",
    "MPLUS1p-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/mplus1p/MPLUS1p-Bold.ttf",
    "MPLUS1-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/mplus1/MPLUS1%5Bwght%5D.ttf",
    "MPLUS2-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/mplus2/MPLUS2%5Bwght%5D.ttf",
    "ZenKakuGothicNew-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/zenkakugothicnew/ZenKakuGothicNew-Regular.ttf",
    "ZenKakuGothicAntique-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/zenkakugothicantique/ZenKakuGothicAntique-Regular.ttf",
    "ZenKakuGothicAntique-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/zenkakugothicantique/ZenKakuGothicAntique-Bold.ttf",
    "BIZUDGothic-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/bizudgothic/BIZUDGothic-Regular.ttf",
    "BIZUDGothic-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/bizudgothic/BIZUDGothic-Bold.ttf",
    "BIZUDPGothic-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/bizudpgothic/BIZUDPGothic-Regular.ttf",
    "IBMPlexSansJP-Thin.ttf": "https://github.com/google/fonts/raw/main/ofl/ibmplexsansjp/IBMPlexSansJP-Thin.ttf",
    "IBMPlexSansJP-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/ibmplexsansjp/IBMPlexSansJP-Regular.ttf",
    "IBMPlexSansJP-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/ibmplexsansjp/IBMPlexSansJP-Bold.ttf",
    "Kosugi-Regular.ttf": "https://github.com/google/fonts/raw/main/apache/kosugi/Kosugi-Regular.ttf",
    "SawarabiGothic-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/sawarabigothic/SawarabiGothic-Regular.ttf",
    "Murecho-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/murecho/Murecho%5Bwght%5D.ttf",
    "DelaGothicOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/delagothicone/DelaGothicOne-Regular.ttf",

    # ===== ARRONDI / ROUNDED =====
    "MPLUSRounded1c-Thin.ttf": "https://github.com/google/fonts/raw/main/ofl/mplusrounded1c/MPLUSRounded1c-Thin.ttf",
    "MPLUSRounded1c-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/mplusrounded1c/MPLUSRounded1c-Bold.ttf",
    "ZenMaruGothic-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/zenmarugothic/ZenMaruGothic-Regular.ttf",
    "ZenMaruGothic-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/zenmarugothic/ZenMaruGothic-Bold.ttf",
    "KiwiMaru-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kiwimaru/KiwiMaru-Regular.ttf",
    "KosugiMaru-Regular.ttf": "https://github.com/google/fonts/raw/main/apache/kosugimaru/KosugiMaru-Regular.ttf",
    "TsukimiRounded-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/tsukimirounded/TsukimiRounded-Regular.ttf",

    # ===== SERIF / MINCHO =====
    "NotoSerifJP-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/notoserifjp/NotoSerifJP%5Bwght%5D.ttf",
    "ZenOldMincho-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/zenoldmincho/ZenOldMincho-Regular.ttf",
    "ZenOldMincho-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/zenoldmincho/ZenOldMincho-Bold.ttf",
    "ShipporiMincho-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/shipporimincho/ShipporiMincho-Regular.ttf",
    "ShipporiMincho-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/shipporimincho/ShipporiMincho-Bold.ttf",
    "ShipporiMinchoB1-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/shipporiminchob1/ShipporiMinchoB1-Regular.ttf",
    "HinaMincho-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/hinamincho/HinaMincho-Regular.ttf",
    "SawarabiMincho-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/sawarabimincho/SawarabiMincho-Regular.ttf",
    "BIZUDMincho-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/bizudmincho/BIZUDMincho-Regular.ttf",
    "BIZUDMincho-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/bizudmincho/BIZUDMincho-Bold.ttf",
    "BIZUDPMincho-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/bizudpmincho/BIZUDPMincho-Regular.ttf",

    # ===== KAISEI (4 familles, serif élégant) =====
    "KaiseiDecol-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseidecol/KaiseiDecol-Regular.ttf",
    "KaiseiDecol-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseidecol/KaiseiDecol-Bold.ttf",
    "KaiseiTokumin-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseitokumin/KaiseiTokumin-Regular.ttf",
    "KaiseiTokumin-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseitokumin/KaiseiTokumin-Bold.ttf",
    "KaiseiOpti-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseiopti/KaiseiOpti-Regular.ttf",
    "KaiseiHarunoUmi-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseiharunoumi/KaiseiHarunoUmi-Regular.ttf",

    # ===== CALLIGRAPHIQUE / ÉDUCATIF =====
    "KleeOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kleeone/KleeOne-Regular.ttf",
    "KleeOne-SemiBold.ttf": "https://github.com/google/fonts/raw/main/ofl/kleeone/KleeOne-SemiBold.ttf",

    # ===== MANUSCRIT / HANDWRITING =====
    "HachiMaruPop-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/hachimarupop/HachiMaruPop-Regular.ttf",
    "YuseiMagic-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/yuseimagic/YuseiMagic-Regular.ttf",
    "PottaOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/pottaone/PottaOne-Regular.ttf",
    "Yomogi-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/yomogi/Yomogi-Regular.ttf",
    "SlacksideOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/slacksideone/SlacksideOne-Regular.ttf",
    "DarumadropOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/darumadropone/DarumadropOne-Regular.ttf",

    # ===== POP / KAWAII =====
    "CherryBombOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/cherrybombone/CherryBombOne-Regular.ttf",
    "MochiyPopOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/mochiypopone/MochiyPopOne-Regular.ttf",
    "MochiyPopPOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/mochiypoppone/MochiyPopPOne-Regular.ttf",

    # ===== PINCEAU / CALLIGRAPHIE =====
    "YujiSyuku-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/yujisyuku/YujiSyuku-Regular.ttf",
    "YujiBoku-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/yujiboku/YujiBoku-Regular.ttf",
    "YujiMai-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/yujimai/YujiMai-Regular.ttf",
    "AoboshiOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/aoboshione/AoboshiOne-Regular.ttf",
    "Chokokutai-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/chokokutai/Chokokutai-Regular.ttf",

    # ===== DISPLAY / FUN =====
    "ReggaeOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/reggaeone/ReggaeOne-Regular.ttf",
    "RocknRollOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/rocknrollone/RocknRollOne-Regular.ttf",
    "WDXLLubrifontJPN-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/wdxllubrifontjpn/WDXLLubrifontJPN-Regular.ttf",

    # ===== PIXEL / MONOSPACE =====
    "Stick-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/stick/Stick-Regular.ttf",
    "DotGothic16-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/dotgothic16/DotGothic16-Regular.ttf",
    "MPLUS1Code-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/mplus1code/MPLUS1Code%5Bwght%5D.ttf",

    # ===== FIN / ÉLÉGANT =====
    "ZenKurenaido-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/zenkurenaido/ZenKurenaido-Regular.ttf",

    # ===== ANTIQUE =====
    "ZenAntique-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/zenantique/ZenAntique-Regular.ttf",
    "ZenAntiqueSoft-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/zenantiquesoft/ZenAntiqueSoft-Regular.ttf",
    "ShipporiAntique-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/shipporiantique/ShipporiAntique-Regular.ttf",
    "ShipporiAntiqueB1-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/shipporiantiqueb1/ShipporiAntiqueB1-Regular.ttf",

    # ===== MODERN / CORPORATE =====
    "LINESeedJP-Thin.ttf": "https://github.com/google/fonts/raw/main/ofl/lineseedjp/LINESeedJP-Thin.ttf",
    "LINESeedJP-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/lineseedjp/LINESeedJP-Regular.ttf",
    "LINESeedJP-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/lineseedjp/LINESeedJP-Bold.ttf",
    "LINESeedJP-ExtraBold.ttf": "https://github.com/google/fonts/raw/main/ofl/lineseedjp/LINESeedJP-ExtraBold.ttf",

    # ===== KANA DÉCORATIF =====
    "Kapakana-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kapakana/Kapakana%5Bwght%5D.ttf",

    # ===== RÉTRO =====
    "NewTegomin-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/newtegomin/NewTegomin-Regular.ttf",
}


def download_font(name: str, url: str) -> None:
    dest = FONTS_DIR / name
    if dest.exists():
        return
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)


def generate_recap(fonts_dir: Path) -> None:
    """Génère un récap visuel : table de toutes les polices avec あ い う え お."""
    PREVIEW_CHARS = ["あ", "い", "う", "え", "お"]
    CELL_SIZE = 48
    FONT_SIZE = 36
    LABEL_WIDTH = 320
    PADDING = 8

    font_files = sorted(fonts_dir.glob("*.ttf"))
    if not font_files:
        print("Aucune police trouvée.")
        return

    # Ajouter les polices système Hiragino
    system_fonts = [
        Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"),
        Path("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"),
        Path("/System/Library/Fonts/ヒラギノ角ゴシック W9.ttc"),
        Path("/System/Library/Fonts/ヒラギノ明朝 ProN.ttc"),
        Path("/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc"),
    ]
    all_fonts = [(f, f.stem) for f in font_files]
    for sf in system_fonts:
        if sf.exists():
            all_fonts.append((sf, f"[système] {sf.stem}"))

    n_fonts = len(all_fonts)
    n_cols = len(PREVIEW_CHARS)

    # Dimensions de l'image
    header_h = CELL_SIZE
    row_h = CELL_SIZE
    img_w = LABEL_WIDTH + n_cols * CELL_SIZE + PADDING * 2
    img_h = header_h + n_fonts * row_h + PADDING * 2

    img = Image.new("RGB", (img_w, img_h), "#1e1e1e")
    draw = ImageDraw.Draw(img)

    # Police pour les labels (système)
    try:
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        label_font = ImageFont.load_default()

    # En-tête
    for j, char in enumerate(PREVIEW_CHARS):
        cx = LABEL_WIDTH + j * CELL_SIZE + CELL_SIZE // 2
        draw.text((cx, PADDING + header_h // 2), char, fill="#888",
                  font=label_font, anchor="mm")

    # Ligne séparatrice
    y_sep = header_h + PADDING
    draw.line([(PADDING, y_sep), (img_w - PADDING, y_sep)], fill="#444", width=1)

    # Lignes par police
    valid_count = 0
    for i, (font_path, font_name) in enumerate(all_fonts):
        y = header_h + PADDING + i * row_h + row_h // 2

        # Label
        display_name = font_name[:40]
        draw.text((PADDING + 4, y), display_name, fill="#ccc",
                  font=label_font, anchor="lm")

        # Rendu des kana
        try:
            font = ImageFont.truetype(str(font_path), FONT_SIZE)
        except Exception:
            draw.text((LABEL_WIDTH + PADDING, y), "✗ erreur", fill="#e74c3c",
                      font=label_font, anchor="lm")
            continue

        rendered = False
        for j, char in enumerate(PREVIEW_CHARS):
            cx = LABEL_WIDTH + j * CELL_SIZE + CELL_SIZE // 2
            try:
                draw.text((cx, y), char, fill="white", font=font, anchor="mm")
                rendered = True
            except Exception:
                pass

        if rendered:
            valid_count += 1

    # Sauvegarder
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    out_path = plots_dir / "fonts_recap.png"
    img.save(str(out_path))

    # Table texte dans le terminal
    print(f"\n{'─' * 60}")
    print(f" RÉCAP : {valid_count}/{n_fonts} polices valides")
    print(f"{'─' * 60}")
    print(f" {'#':>3s}  {'Police':<42s}  {'Statut'}")
    print(f"{'─' * 60}")
    for i, (font_path, font_name) in enumerate(all_fonts):
        try:
            ImageFont.truetype(str(font_path), FONT_SIZE)
            status = "✓"
        except Exception:
            status = "✗"
        prefix = "[sys]" if "[système]" in font_name else "     "
        clean_name = font_name.replace("[système] ", "")
        print(f" {i + 1:3d}  {prefix} {clean_name:<36s}  {status}")
    print(f"{'─' * 60}")
    print(f"\n Aperçu visuel sauvé → {out_path}")


def main() -> None:
    FONTS_DIR.mkdir(parents=True, exist_ok=True)

    # Supprimer les anciennes polices creuses si présentes
    for hollow in ["RampartOne-Regular.ttf", "TrainOne-Regular.ttf"]:
        p = FONTS_DIR / hollow
        if p.exists():
            p.unlink()
            print(f"  Supprimé (creux) : {hollow}")

    print(f"Téléchargement de {len(GOOGLE_FONTS)} polices japonaises...")
    for name, url in tqdm(GOOGLE_FONTS.items(), desc="  Fonts"):
        try:
            download_font(name, url)
        except Exception as e:
            print(f"\n  ⚠ {name}: {e}")

    n_downloaded = len(list(FONTS_DIR.glob("*.ttf")))
    print(f"\n{n_downloaded} polices dans {FONTS_DIR}/")

    # Récap visuel
    generate_recap(FONTS_DIR)


if __name__ == "__main__":
    main()
