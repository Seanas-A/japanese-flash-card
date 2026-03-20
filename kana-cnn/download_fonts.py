"""Télécharge des polices japonaises gratuites depuis Google Fonts pour enrichir le dataset.

Styles variés : serif, sans-serif, arrondi, calligraphique, manuscrit, pinceau, pixel, gras, fin, etc.
Toutes les polices sont PLEINES (pas de contour/outline).
"""

from pathlib import Path

import requests
from tqdm import tqdm

FONTS_DIR = Path(__file__).parent / "data" / "fonts"

# Polices Google Fonts avec kana — sélectionnées pour la variété de style
# Toutes pleines (pas d'outline/creux)
GOOGLE_FONTS = {
    # --- Sans-serif ---
    "NotoSansJP-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf",
    "NotoSerifJP-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/notoserifjp/NotoSerifJP%5Bwght%5D.ttf",

    # --- M+ Rounded (arrondi, 4 poids) ---
    "MPLUSRounded1c-Thin.ttf": "https://github.com/google/fonts/raw/main/ofl/mplusrounded1c/MPLUSRounded1c-Thin.ttf",
    "MPLUSRounded1c-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/mplusrounded1c/MPLUSRounded1c-Regular.ttf",
    "MPLUSRounded1c-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/mplusrounded1c/MPLUSRounded1c-Bold.ttf",
    "MPLUSRounded1c-Black.ttf": "https://github.com/google/fonts/raw/main/ofl/mplusrounded1c/MPLUSRounded1c-Black.ttf",

    # --- M+ 1p (sans-serif droit, 4 poids) ---
    "MPLUS1p-Thin.ttf": "https://github.com/google/fonts/raw/main/ofl/mplus1p/MPLUS1p-Thin.ttf",
    "MPLUS1p-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/mplus1p/MPLUS1p-Regular.ttf",
    "MPLUS1p-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/mplus1p/MPLUS1p-Bold.ttf",
    "MPLUS1p-Black.ttf": "https://github.com/google/fonts/raw/main/ofl/mplus1p/MPLUS1p-Black.ttf",

    # --- Zen Kaku Gothic (modern, 2 poids) ---
    "ZenKakuGothicNew-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/zenkakugothicnew/ZenKakuGothicNew-Regular.ttf",
    "ZenKakuGothicNew-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/zenkakugothicnew/ZenKakuGothicNew-Bold.ttf",

    # --- Zen Maru Gothic (arrondi doux, 3 poids) ---
    "ZenMaruGothic-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/zenmarugothic/ZenMaruGothic-Regular.ttf",
    "ZenMaruGothic-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/zenmarugothic/ZenMaruGothic-Bold.ttf",
    "ZenMaruGothic-Black.ttf": "https://github.com/google/fonts/raw/main/ofl/zenmarugothic/ZenMaruGothic-Black.ttf",

    # --- Serif classique ---
    "ZenOldMincho-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/zenoldmincho/ZenOldMincho-Regular.ttf",
    "ZenOldMincho-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/zenoldmincho/ZenOldMincho-Bold.ttf",
    "ShipporiMincho-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/shipporimincho/ShipporiMincho-Regular.ttf",
    "ShipporiMincho-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/shipporimincho/ShipporiMincho-Bold.ttf",
    "ShipporiMinchoB1-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/shipporiminchob1/ShipporiMinchoB1-Regular.ttf",
    "ShipporiMinchoB1-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/shipporiminchob1/ShipporiMinchoB1-Bold.ttf",

    # --- Calligraphique / éducatif ---
    "KleeOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kleeone/KleeOne-Regular.ttf",
    "KleeOne-SemiBold.ttf": "https://github.com/google/fonts/raw/main/ofl/kleeone/KleeOne-SemiBold.ttf",

    # --- Manuscrit / kawaii ---
    "HachiMaruPop-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/hachimarupop/HachiMaruPop-Regular.ttf",
    "YuseiMagic-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/yuseimagic/YuseiMagic-Regular.ttf",
    "PottaOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/pottaone/PottaOne-Regular.ttf",

    # --- Pinceau / calligraphie ---
    "YujiSyuku-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/yujisyuku/YujiSyuku-Regular.ttf",
    "YujiBoku-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/yujiboku/YujiBoku-Regular.ttf",
    "YujiMai-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/yujimai/YujiMai-Regular.ttf",

    # --- Display / fun ---
    "DelaGothicOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/delagothicone/DelaGothicOne-Regular.ttf",
    "ReggaeOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/reggaeone/ReggaeOne-Regular.ttf",
    "RocknRollOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/rocknrollone/RocknRollOne-Regular.ttf",

    # --- Pixel / monospace ---
    "Stick-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/stick/Stick-Regular.ttf",
    "DotGothic16-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/dotgothic16/DotGothic16-Regular.ttf",

    # --- Fin / élégant ---
    "ZenKurenaido-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/zenkurenaido/ZenKurenaido-Regular.ttf",
    "SawarabiMincho-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/sawarabimincho/SawarabiMincho-Regular.ttf",
    "SawarabiGothic-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/sawarabigothic/SawarabiGothic-Regular.ttf",

    # --- Zen Antique ---
    "ZenAntique-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/zenantique/ZenAntique-Regular.ttf",
    "ZenAntiqueSoft-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/zenantiquesoft/ZenAntiqueSoft-Regular.ttf",

    # --- Shippori Antique ---
    "ShipporiAntique-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/shipporiantique/ShipporiAntique-Regular.ttf",
    "ShipporiAntiqueB1-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/shipporiantiqueb1/ShipporiAntiqueB1-Regular.ttf",

    # --- Kaisei ---
    "KaiseiDecol-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseidecol/KaiseiDecol-Regular.ttf",
    "KaiseiDecol-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseidecol/KaiseiDecol-Bold.ttf",
    "KaiseiTokumin-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseitokumin/KaiseiTokumin-Regular.ttf",
    "KaiseiTokumin-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseitokumin/KaiseiTokumin-Bold.ttf",
    "KaiseiOpti-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseiopti/KaiseiOpti-Regular.ttf",
    "KaiseiOpti-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseiopti/KaiseiOpti-Bold.ttf",
    "KaiseiHarunoUmi-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseiharunoumi/KaiseiHarunoUmi-Regular.ttf",
    "KaiseiHarunoUmi-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseiharunoumi/KaiseiHarunoUmi-Bold.ttf",

    # --- Murecho (variable weight) ---
    "Murecho-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/murecho/Murecho%5Bwght%5D.ttf",

    # --- Kiwi Maru (arrondi doux) ---
    "KiwiMaru-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kiwimaru/KiwiMaru-Regular.ttf",
    "KiwiMaru-Medium.ttf": "https://github.com/google/fonts/raw/main/ofl/kiwimaru/KiwiMaru-Medium.ttf",

    # --- New Tegomin (rétro) ---
    "NewTegomin-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/newtegomin/NewTegomin-Regular.ttf",
}


def download_font(name: str, url: str) -> None:
    dest = FONTS_DIR / name
    if dest.exists():
        return
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)


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

    print(f"\n{len(list(FONTS_DIR.glob('*.ttf')))} polices dans {FONTS_DIR}/")


if __name__ == "__main__":
    main()
