"""Prépare le dataset de reconnaissance de caractères japonais.

Pipeline complet :
  1. Extraction des 2136 Jōyō kanji (kanjidic2.xml, EDRDG)
  2. Découverte et téléchargement des polices japonaises (Google Fonts)
  3. Rendu des caractères (96 kana + 2136 kanji = 2232 classes)
  4. Sauvegarde (npz + labels.json + split_indices.npz)

Usage :
  uv run python prepare_dataset.py                 # pipeline complet (2232 classes)
  uv run python prepare_dataset.py --kana-only      # 96 classes kana uniquement
  uv run python prepare_dataset.py --review          # page HTML interactive de sélection
  uv run python prepare_dataset.py --refresh-fonts   # re-découvrir polices Google Fonts
"""

import argparse
import base64
import gzip
import hashlib
import json
import os
import platform
import re
import subprocess
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ─── Constantes ──────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
FONTS_DIR = DATA_DIR / "fonts"
FONT_INDEX = FONTS_DIR / "font_index.json"
FONT_SELECTION_FILE = DATA_DIR / "font_selection.json"
PLOTS_DIR = Path(__file__).parent / "plots"
SEED = 42
IMG_SIZE = 28
RENDER_SIZE = 64
CANVAS_SIZE = 96

# ─── Kana ────────────────────────────────────────────────────────────
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


# =====================================================================
# Jōyō kanji (kanjidic2.xml — EDRDG, licence CC BY-SA 4.0)
# =====================================================================

KANJIDIC2_URL = "http://www.edrdg.org/kanjidic/kanjidic2.xml.gz"
JOYO_CACHE = DATA_DIR / "joyo_kanji.json"


def load_joyo_kanji() -> list[dict]:
    """Charge les 2136 Jōyō kanji (cache local ou téléchargement kanjidic2)."""
    if JOYO_CACHE.exists():
        with open(JOYO_CACHE, encoding="utf-8") as f:
            data = json.load(f)
        if len(data) >= 2100:
            return data

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Téléchargement de kanjidic2.xml.gz...")
    resp = requests.get(KANJIDIC2_URL, timeout=60)
    resp.raise_for_status()
    root = ET.parse(BytesIO(gzip.decompress(resp.content))).getroot()

    joyo = []
    for char_el in root.iter("character"):
        misc = char_el.find("misc")
        if misc is None:
            continue
        grade_el = misc.find("grade")
        if grade_el is None:
            continue
        grade = int(grade_el.text)
        if grade not in (1, 2, 3, 4, 5, 6, 8):
            continue

        literal = char_el.find("literal").text
        on_reading, kun_reading = "", ""
        rmgroup = char_el.find("reading_meaning")
        if rmgroup is not None:
            rmgroup = rmgroup.find("rmgroup")
        if rmgroup is not None:
            for r in rmgroup.findall("reading"):
                if r.get("r_type") == "ja_on" and not on_reading:
                    on_reading = r.text or ""
                elif r.get("r_type") == "ja_kun" and not kun_reading:
                    kun_reading = r.text or ""

        joyo.append({"char": literal, "grade": grade,
                     "on_reading": on_reading, "kun_reading": kun_reading})

    joyo.sort(key=lambda k: (k["grade"], ord(k["char"])))
    with open(JOYO_CACHE, "w", encoding="utf-8") as f:
        json.dump(joyo, f, ensure_ascii=False, indent=2)
    print(f"  {len(joyo)} Jōyō kanji extraits → {JOYO_CACHE.name}")
    return joyo


# =====================================================================
# Polices Google Fonts — fallback statique
# =====================================================================

FALLBACK_FONTS = {
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
    "MPLUSRounded1c-Thin.ttf": "https://github.com/google/fonts/raw/main/ofl/mplusrounded1c/MPLUSRounded1c-Thin.ttf",
    "MPLUSRounded1c-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/mplusrounded1c/MPLUSRounded1c-Bold.ttf",
    "ZenMaruGothic-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/zenmarugothic/ZenMaruGothic-Regular.ttf",
    "ZenMaruGothic-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/zenmarugothic/ZenMaruGothic-Bold.ttf",
    "KiwiMaru-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kiwimaru/KiwiMaru-Regular.ttf",
    "KosugiMaru-Regular.ttf": "https://github.com/google/fonts/raw/main/apache/kosugimaru/KosugiMaru-Regular.ttf",
    "TsukimiRounded-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/tsukimirounded/TsukimiRounded-Regular.ttf",
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
    "KaiseiDecol-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseidecol/KaiseiDecol-Regular.ttf",
    "KaiseiDecol-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseidecol/KaiseiDecol-Bold.ttf",
    "KaiseiTokumin-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseitokumin/KaiseiTokumin-Regular.ttf",
    "KaiseiTokumin-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseitokumin/KaiseiTokumin-Bold.ttf",
    "KaiseiOpti-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseiopti/KaiseiOpti-Regular.ttf",
    "KaiseiHarunoUmi-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kaiseiharunoumi/KaiseiHarunoUmi-Regular.ttf",
    "KleeOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kleeone/KleeOne-Regular.ttf",
    "KleeOne-SemiBold.ttf": "https://github.com/google/fonts/raw/main/ofl/kleeone/KleeOne-SemiBold.ttf",
    "HachiMaruPop-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/hachimarupop/HachiMaruPop-Regular.ttf",
    "YuseiMagic-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/yuseimagic/YuseiMagic-Regular.ttf",
    "PottaOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/pottaone/PottaOne-Regular.ttf",
    "Yomogi-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/yomogi/Yomogi-Regular.ttf",
    "SlacksideOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/slacksideone/SlacksideOne-Regular.ttf",
    "DarumadropOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/darumadropone/DarumadropOne-Regular.ttf",
    "CherryBombOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/cherrybombone/CherryBombOne-Regular.ttf",
    "MochiyPopOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/mochiypopone/MochiyPopOne-Regular.ttf",
    "MochiyPopPOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/mochiypoppone/MochiyPopPOne-Regular.ttf",
    "YujiSyuku-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/yujisyuku/YujiSyuku-Regular.ttf",
    "YujiBoku-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/yujiboku/YujiBoku-Regular.ttf",
    "YujiMai-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/yujimai/YujiMai-Regular.ttf",
    "AoboshiOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/aoboshione/AoboshiOne-Regular.ttf",
    "Chokokutai-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/chokokutai/Chokokutai-Regular.ttf",
    "ReggaeOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/reggaeone/ReggaeOne-Regular.ttf",
    "RocknRollOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/rocknrollone/RocknRollOne-Regular.ttf",
    "WDXLLubrifontJPN-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/wdxllubrifontjpn/WDXLLubrifontJPN-Regular.ttf",
    "Stick-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/stick/Stick-Regular.ttf",
    "DotGothic16-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/dotgothic16/DotGothic16-Regular.ttf",
    "MPLUS1Code-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/mplus1code/MPLUS1Code%5Bwght%5D.ttf",
    "ZenKurenaido-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/zenkurenaido/ZenKurenaido-Regular.ttf",
    "ZenAntique-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/zenantique/ZenAntique-Regular.ttf",
    "ZenAntiqueSoft-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/zenantiquesoft/ZenAntiqueSoft-Regular.ttf",
    "ShipporiAntique-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/shipporiantique/ShipporiAntique-Regular.ttf",
    "ShipporiAntiqueB1-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/shipporiantiqueb1/ShipporiAntiqueB1-Regular.ttf",
    "LINESeedJP-Thin.ttf": "https://github.com/google/fonts/raw/main/ofl/lineseedjp/LINESeedJP-Thin.ttf",
    "LINESeedJP-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/lineseedjp/LINESeedJP-Regular.ttf",
    "LINESeedJP-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/lineseedjp/LINESeedJP-Bold.ttf",
    "LINESeedJP-ExtraBold.ttf": "https://github.com/google/fonts/raw/main/ofl/lineseedjp/LINESeedJP-ExtraBold.ttf",
    "Kapakana-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/kapakana/Kapakana%5Bwght%5D.ttf",
    "NewTegomin-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/newtegomin/NewTegomin-Regular.ttf",
}


# =====================================================================
# Découverte et téléchargement des polices
# =====================================================================

def discover_japanese_fonts() -> dict[str, str]:
    """Découvre les polices japonaises via le catalogue Google Fonts."""
    if FONT_INDEX.exists():
        with open(FONT_INDEX, encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) >= len(FALLBACK_FONTS):
            return cached

    print("Découverte des polices japonaises sur Google Fonts...")
    try:
        resp = requests.get("https://fonts.google.com/metadata/fonts", timeout=30,
                            headers={"User-Agent": "kana-cnn-dataset/1.0"})
        resp.raise_for_status()
        text = resp.text
        if text.startswith(")]}'"):
            text = text[4:].lstrip("\n")

        families = json.loads(text).get("familyMetadataList", [])
        jp_families = [f for f in families if "japanese" in f.get("subsets", [])]
        print(f"  {len(jp_families)} familles japonaises trouvées")

        fonts = {}
        for fam in jp_families:
            dir_name = fam.get("family", "").lower().replace(" ", "")
            fonts.update(_resolve_family_fonts(dir_name))

        for name, url in FALLBACK_FONTS.items():
            fonts.setdefault(name, url)

        FONTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(FONT_INDEX, "w", encoding="utf-8") as f:
            json.dump(fonts, f, indent=2)
        print(f"  {len(fonts)} polices indexées")
        return fonts

    except Exception as e:
        print(f"  Découverte échouée ({e}), fallback ({len(FALLBACK_FONTS)} polices)")
        return dict(FALLBACK_FONTS)


def _resolve_family_fonts(dir_name: str) -> dict[str, str]:
    """Résout les URLs .ttf pour une famille depuis le repo google/fonts."""
    base = "https://raw.githubusercontent.com/google/fonts/main"
    fonts = {}
    for lic in ("ofl", "apache"):
        try:
            resp = requests.get(f"{base}/{lic}/{dir_name}/METADATA.pb", timeout=10)
            if resp.status_code != 200:
                continue
            for fname in re.findall(r'filename:\s*"([^"]+\.ttf)"', resp.text):
                url_fname = fname.replace("[", "%5B").replace("]", "%5D")
                local = fname.replace("[wght]", "-Regular")
                fonts[local] = f"https://github.com/google/fonts/raw/main/{lic}/{dir_name}/{url_fname}"
            if fonts:
                break
        except Exception:
            continue
    return fonts


def download_all_fonts(font_list: dict[str, str], workers: int = 8) -> int:
    """Télécharge les polices manquantes en parallèle."""
    FONTS_DIR.mkdir(parents=True, exist_ok=True)

    def _dl(item):
        name, url = item
        dest = FONTS_DIR / name
        if dest.exists():
            return False
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return True
        except Exception:
            return False

    new = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ok in tqdm(pool.map(_dl, font_list.items()),
                       total=len(font_list), desc="  Téléchargement", unit="font"):
            if ok:
                new += 1
    return new


def collect_fonts() -> list[tuple[str, str]]:
    """Collecte les polices valides, dédupliquées par rendu visuel."""
    font_paths = [(p.stem, str(p)) for p in sorted(FONTS_DIR.glob("*.ttf"))]

    # Polices système
    if platform.system() == "Darwin":
        for p in [f"/System/Library/Fonts/ヒラギノ角ゴシック W{i}.ttc" for i in range(10)]:
            if os.path.exists(p):
                font_paths.append((os.path.basename(p).replace(".ttc", ""), p))
    else:
        try:
            res = subprocess.run(["fc-list", ":lang=ja", "file"],
                                 capture_output=True, text=True, timeout=10)
            for line in res.stdout.strip().split("\n"):
                p = line.strip().rstrip(":")
                if p and os.path.exists(p):
                    font_paths.append((Path(p).stem, p))
        except Exception:
            pass

    # Dédupliquer par hash du rendu de あ
    valid, seen = [], set()
    for name, path in font_paths:
        try:
            font = ImageFont.truetype(path, RENDER_SIZE)
            img = _render_char(font, "あ")
            if img is None:
                continue
            h = hashlib.md5(img.tobytes()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            valid.append((name, path))
        except Exception:
            continue
    return valid


def load_font_selection(fonts: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Filtre les polices selon font_selection.json si présent."""
    if not FONT_SELECTION_FILE.exists():
        return fonts
    with open(FONT_SELECTION_FILE, encoding="utf-8") as f:
        sel = {e["name"]: e["keep"] for e in json.load(f)}
    kept = [(n, p) for n, p in fonts if sel.get(n, True)]
    excluded = len(fonts) - len(kept)
    if excluded:
        print(f"  {excluded} polices exclues par font_selection.json")
    return kept


# =====================================================================
# Rendu des caractères
# =====================================================================

def _render_char(font: ImageFont.FreeTypeFont, char: str) -> np.ndarray | None:
    """Rend un caractère centré, fond noir, tracé blanc, 28x28."""
    img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if w < 3 or h < 3:
        return None
    x = (CANVAS_SIZE - w) // 2 - bbox[0]
    y = (CANVAS_SIZE - h) // 2 - bbox[1]
    draw.text((x, y), char, fill=255, font=font)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.array(img)


def _get_cmap(font_path: str) -> set[int]:
    """Extrait les codepoints supportés par la police via fontTools."""
    try:
        ttfont = TTFont(font_path, fontNumber=0)
        cmap = ttfont.getBestCmap()
        ttfont.close()
        return set(cmap.keys()) if cmap else set()
    except Exception:
        return set()


def _render_font_chars(font_path: str, characters: list[tuple[str, int]],
                       ) -> list[tuple[np.ndarray, int]]:
    """Rend les caractères supportés par une police.

    Filtrage :
      1. cmap fontTools — rejette les codepoints absents (évite le fallback Pillow)
      2. hash tofu — rejette les glyphes identiques au placeholder .notdef
    """
    cmap = _get_cmap(font_path)
    try:
        font = ImageFont.truetype(font_path, RENDER_SIZE)
    except Exception:
        return []

    # Calculer le hash du tofu (.notdef) via un codepoint garanti absent
    tofu_hash = None
    tofu_img = _render_char(font, "\uffff")
    if tofu_img is not None:
        tofu_hash = hashlib.md5(tofu_img.tobytes()).hexdigest()

    results = []
    for char, label_idx in characters:
        if ord(char) not in cmap:
            continue
        img = _render_char(font, char)
        if img is None:
            continue
        if tofu_hash and hashlib.md5(img.tobytes()).hexdigest() == tofu_hash:
            continue
        results.append((img, label_idx))
    return results


def build_character_list(kana_only: bool = False) -> list[tuple[str, int]]:
    """Construit la liste (char, label_index). Layout: 0-47 hira, 48-95 kata, 96+ kanji."""
    chars = []
    for i, (c, _) in enumerate(HIRAGANA):
        chars.append((c, i))
    for i, (c, _) in enumerate(KATAKANA):
        chars.append((c, 48 + i))
    if not kana_only:
        for i, k in enumerate(load_joyo_kanji()):
            chars.append((k["char"], 96 + i))
    return chars


def render_all(fonts: list[tuple[str, str]], characters: list[tuple[str, int]],
               ) -> tuple[np.ndarray, np.ndarray]:
    """Rend tous les caractères dans toutes les polices."""
    all_images, all_labels = [], []
    desc = f"  Rendu ({len(fonts)} polices × {len(characters)} chars)"
    for _, path in tqdm(fonts, desc=desc, unit="font"):
        for img, label in _render_font_chars(path, characters):
            all_images.append(img)
            all_labels.append(label)

    images = np.array(all_images, dtype=np.uint8)
    max_label = max(all_labels) if all_labels else 0
    labels = np.array(all_labels, dtype=np.uint8 if max_label < 256 else np.uint16)
    return images, labels


# =====================================================================
# Sauvegarde
# =====================================================================

def generate_labels_json(kana_only: bool = False) -> int:
    """Génère labels.json. Retourne le nombre de classes."""
    labels = []
    for i, (c, r) in enumerate(HIRAGANA):
        labels.append({"index": i, "char": c, "romaji": r, "type": "hiragana"})
    for i, (c, r) in enumerate(KATAKANA):
        labels.append({"index": 48 + i, "char": c, "romaji": r, "type": "katakana"})
    if not kana_only:
        for i, k in enumerate(load_joyo_kanji()):
            labels.append({"index": 96 + i, "char": k["char"],
                           "on_reading": k["on_reading"], "kun_reading": k["kun_reading"],
                           "grade": k["grade"], "type": "kanji"})
    with open(DATA_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    return len(labels)


def save_dataset(images: np.ndarray, labels: np.ndarray) -> None:
    """Sauvegarde le dataset compressé."""
    out = DATA_DIR / "kana_dataset.npz"
    np.savez_compressed(out, images=images, labels=labels)
    print(f"  → {out.name} ({out.stat().st_size / 1024**2:.1f} MB)")


def save_splits(labels: np.ndarray) -> None:
    """Splits train/val/test stratifiés 70/15/15."""
    idx = np.arange(len(labels))
    train, temp = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=labels)
    val, test = train_test_split(temp, test_size=0.50, random_state=SEED, stratify=labels[temp])
    np.savez(DATA_DIR / "split_indices.npz", train_idx=train, val_idx=val, test_idx=test)

    print(f"\nSplit (seed={SEED}) :")
    for name, si in [("Train", train), ("Val", val), ("Test", test)]:
        sl = labels[si]
        h = int(np.sum(sl < 48))
        k = int(np.sum((sl >= 48) & (sl < 96)))
        kj = int(np.sum(sl >= 96))
        parts = f"{h} hira + {k} kata"
        if kj:
            parts += f" + {kj} kanji"
        print(f"  {name:5s}: {len(si):6d} ({len(si)/len(labels)*100:.0f}%) — {parts}")


# =====================================================================
# Review HTML interactif — aperçu exhaustif avec 🛑 pour les absents
# =====================================================================

# Caractères variés pour l'aperçu
REVIEW_HIRAGANA = ["あ", "き", "ふ", "を", "ん"]
REVIEW_KATAKANA = ["ア", "シ", "ツ", "ヨ", "ン"]
REVIEW_KANJI = ["一", "語", "飛", "鬱", "𠮟", "剝", "塡", "頰", "書", "読"]


def _render_preview_b64(font: ImageFont.FreeTypeFont, char: str) -> str | None:
    """Rend un caractère en base64 PNG pour le HTML."""
    canvas = int(RENDER_SIZE * 1.5)
    img = Image.new("L", (canvas, canvas), 0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if w < 2 or h < 2:
        return None
    x = (canvas - w) // 2 - bbox[0]
    y = (canvas - h) // 2 - bbox[1]
    draw.text((x, y), char, fill=255, font=font)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def generate_review_html(fonts: list[tuple[str, str]]) -> Path:
    """Génère une page HTML avec aperçu de chaque police.

    - Caractère rendu normalement si supporté (dans le cmap)
    - 🛑 si absent du cmap (= serait exclu du dataset)
    - Toggle pour garder/exclure chaque police
    """
    chars = REVIEW_HIRAGANA + REVIEW_KATAKANA + REVIEW_KANJI

    existing_sel = {}
    if FONT_SELECTION_FILE.exists():
        with open(FONT_SELECTION_FILE, encoding="utf-8") as f:
            existing_sel = {e["name"]: e["keep"] for e in json.load(f)}

    print("  Rendu des aperçus...")
    rows = []
    for font_name, font_path in tqdm(fonts, desc="  Aperçus", unit="font"):
        cmap = _get_cmap(font_path)
        try:
            font = ImageFont.truetype(font_path, RENDER_SIZE)
        except Exception:
            continue

        # Hash du tofu pour cette police
        tofu_hash = None
        tofu_img = _render_char(font, "\uffff")
        if tofu_img is not None:
            tofu_hash = hashlib.md5(tofu_img.tobytes()).hexdigest()

        previews = {}  # char → b64 | "STOP"
        n_supported = 0
        for char in chars:
            if ord(char) not in cmap:
                previews[char] = "STOP"
                continue
            b64 = _render_preview_b64(font, char)
            if b64 is None:
                previews[char] = "STOP"
                continue
            # Vérifier tofu via hash
            rendered = _render_char(font, char)
            if rendered is not None and tofu_hash:
                if hashlib.md5(rendered.tobytes()).hexdigest() == tofu_hash:
                    previews[char] = "STOP"
                    continue
            previews[char] = b64
            n_supported += 1

        # Couverture totale (tous les caractères, pas juste les preview)
        all_kana = [c for c, _ in HIRAGANA + KATAKANA]
        all_kanji = [k["char"] for k in load_joyo_kanji()]
        kana_cov = sum(1 for c in all_kana if ord(c) in cmap)
        kanji_cov = sum(1 for c in all_kanji if ord(c) in cmap)

        keep = existing_sel.get(font_name, True)
        rows.append({"name": font_name, "previews": previews, "keep": keep,
                      "kana_cov": kana_cov, "kanji_cov": kanji_cov})

    # HTML
    header_cells = ""
    for c in chars:
        if c in REVIEW_HIRAGANA:
            cls = "hira"
        elif c in REVIEW_KATAKANA:
            cls = "kata"
        else:
            cls = "kanji"
        header_cells += f'<th class="char-header {cls}">{c}</th>'

    table_rows = []
    for i, row in enumerate(rows):
        checked = "checked" if row["keep"] else ""
        cells = f'<td class="font-name">{row["name"]}</td>'
        cells += f'<td class="cov">{row["kana_cov"]}/96</td>'
        cells += f'<td class="cov">{row["kanji_cov"]}/2136</td>'
        for char in chars:
            val = row["previews"].get(char, "STOP")
            if val == "STOP":
                cells += '<td class="stop">🛑</td>'
            else:
                cells += f'<td><img src="data:image/png;base64,{val}" width="48" height="48"></td>'
        cells += (f'<td class="toggle-cell"><label class="switch">'
                  f'<input type="checkbox" data-font="{row["name"]}" {checked}>'
                  f'<span class="slider"></span></label></td>')
        table_rows.append(f'<tr data-idx="{i}">{cells}</tr>')

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Review des polices — Kana CNN</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #1a1a2e; color: #eaeaea; font-family: -apple-system, 'Segoe UI', sans-serif; padding: 20px; }}
h1 {{ text-align: center; margin-bottom: 8px; font-size: 1.5em; }}
.subtitle {{ text-align: center; color: #888; margin-bottom: 20px; font-size: 0.9em; }}
.toolbar {{ display: flex; justify-content: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
.toolbar button {{
    padding: 8px 20px; border: none; border-radius: 6px; cursor: pointer;
    font-size: 0.9em; font-weight: 600; transition: background 0.2s;
}}
.btn-save {{ background: #2ecc71; color: #fff; }}
.btn-save:hover {{ background: #27ae60; }}
.btn-all {{ background: #3498db; color: #fff; }}
.btn-all:hover {{ background: #2980b9; }}
.btn-none {{ background: #e74c3c; color: #fff; }}
.btn-none:hover {{ background: #c0392b; }}
.btn-invert {{ background: #9b59b6; color: #fff; }}
.btn-invert:hover {{ background: #8e44ad; }}
.stats {{ text-align: center; margin-bottom: 16px; font-size: 0.95em; color: #aaa; }}
.stats strong {{ color: #2ecc71; }}
table {{ border-collapse: collapse; width: 100%; background: #16213e; border-radius: 8px; overflow: hidden; }}
thead {{ position: sticky; top: 0; z-index: 10; }}
th {{ background: #0f3460; padding: 10px 6px; font-size: 0.8em; white-space: nowrap; }}
th.hira {{ color: #e94560; }}
th.kata {{ color: #4ea8de; }}
th.kanji {{ color: #2ecc71; }}
td {{ padding: 4px; text-align: center; border-bottom: 1px solid #1a1a2e; vertical-align: middle; }}
td img {{ display: block; margin: auto; image-rendering: pixelated; border-radius: 4px; }}
td.font-name {{ text-align: left; padding-left: 12px; font-size: 0.85em; white-space: nowrap; max-width: 220px; overflow: hidden; text-overflow: ellipsis; }}
td.cov {{ font-size: 0.75em; color: #888; white-space: nowrap; }}
td.stop {{ font-size: 1.4em; }}
tr {{ transition: background 0.15s; }}
tr:hover {{ background: #1e2d50; }}
tr.excluded {{ opacity: 0.35; }}
.toggle-cell {{ padding: 4px 12px; }}
.switch {{ position: relative; display: inline-block; width: 44px; height: 24px; }}
.switch input {{ opacity: 0; width: 0; height: 0; }}
.slider {{ position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
           background: #555; border-radius: 24px; transition: 0.3s; }}
.slider:before {{ content: ""; position: absolute; height: 18px; width: 18px; left: 3px; bottom: 3px;
                  background: white; border-radius: 50%; transition: 0.3s; }}
input:checked + .slider {{ background: #2ecc71; }}
input:checked + .slider:before {{ transform: translateX(20px); }}
</style>
</head>
<body>
<h1>Review des polices</h1>
<p class="subtitle">🛑 = caractère absent du cmap (exclu du dataset). Décochez les polices à exclure.</p>
<div class="toolbar">
    <button class="btn-save" onclick="save()">Sauvegarder</button>
    <button class="btn-all" onclick="setAll(true)">Tout cocher</button>
    <button class="btn-none" onclick="setAll(false)">Tout décocher</button>
    <button class="btn-invert" onclick="invert()">Inverser</button>
</div>
<div class="stats" id="stats"></div>
<table>
<thead><tr>
    <th>Police</th><th>Kana</th><th>Kanji</th>
    {header_cells}
    <th>Garder</th>
</tr></thead>
<tbody>
{"".join(table_rows)}
</tbody>
</table>
<script>
const total = {len(rows)};
function updateStats() {{
    const n = document.querySelectorAll('input[type=checkbox]:checked').length;
    document.getElementById('stats').innerHTML = `<strong>${{n}}</strong> / ${{total}} polices`;
    document.querySelectorAll('tbody tr').forEach(tr => {{
        tr.classList.toggle('excluded', !tr.querySelector('input[type=checkbox]').checked);
    }});
}}
function setAll(v) {{ document.querySelectorAll('input[type=checkbox]').forEach(c => c.checked = v); updateStats(); }}
function invert() {{ document.querySelectorAll('input[type=checkbox]').forEach(c => c.checked = !c.checked); updateStats(); }}
function save() {{
    const data = [];
    document.querySelectorAll('input[type=checkbox]').forEach(c => {{
        data.push({{ name: c.dataset.font, keep: c.checked }});
    }});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {{ type: 'application/json' }}));
    a.download = 'font_selection.json';
    a.click();
    alert('Fichier telecharge !\\nPlacez-le dans kana-cnn/data/font_selection.json');
}}
document.querySelectorAll('input[type=checkbox]').forEach(c => c.addEventListener('change', updateStats));
updateStats();
</script>
</body>
</html>"""

    out = PLOTS_DIR / "font_review.html"
    PLOTS_DIR.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"  → {out}")
    return out


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset de caractères japonais.")
    parser.add_argument("--refresh-fonts", action="store_true",
                        help="Re-découvrir les polices depuis Google Fonts")
    parser.add_argument("--kana-only", action="store_true",
                        help="96 classes kana uniquement")
    parser.add_argument("--workers", type=int, default=8,
                        help="Threads de téléchargement")
    parser.add_argument("--review", action="store_true",
                        help="Page HTML interactive pour sélectionner les polices")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Polices ──
    print("=" * 60)
    print(" PHASE 1 : Polices")
    print("=" * 60)
    if args.refresh_fonts and FONT_INDEX.exists():
        FONT_INDEX.unlink()
    font_list = discover_japanese_fonts()
    new = download_all_fonts(font_list, workers=args.workers)
    print(f"  {new} nouvelles polices téléchargées")
    fonts = collect_fonts()
    print(f"  {len(fonts)} polices uniques au total")

    if args.review:
        html_path = generate_review_html(fonts)
        print(f"\n  Ouvrez {html_path} dans votre navigateur.")
        print(f"  Placez le fichier sauvegardé dans {FONT_SELECTION_FILE}")
        return

    fonts = load_font_selection(fonts)
    print(f"  {len(fonts)} polices retenues pour le dataset")

    # ── Rendu ──
    print("\n" + "=" * 60)
    mode = "kana (96 classes)" if args.kana_only else "kana + kanji (2232 classes)"
    print(f" PHASE 2 : Rendu — {mode}")
    print("=" * 60)
    characters = build_character_list(kana_only=args.kana_only)
    images, labels = render_all(fonts, characters)
    print(f"\n  {len(images)} images générées")
    print(f"  {len(np.unique(labels))} classes couvertes")

    # ── Sauvegarde ──
    print("\n" + "=" * 60)
    print(" PHASE 3 : Sauvegarde")
    print("=" * 60)
    save_dataset(images, labels)
    save_splits(labels)
    n = generate_labels_json(kana_only=args.kana_only)
    print(f"  labels.json : {n} classes")
    print("\nTerminé.")


if __name__ == "__main__":
    main()
