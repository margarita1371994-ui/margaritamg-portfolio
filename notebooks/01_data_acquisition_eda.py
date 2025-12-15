"""
# 01 — Ingestion & Feature Engineering (CARBOSOL + AEMET) + EDA

**Objetivo**
  - Construir un dataset a nivel **perfil** uniendo:
  - CARBOSOL profiles + horizons
  - Variable objetivo a partir de **Description → cultivo**
  - Agrupación avanzada de cultivo (**cultivo_grupo**)
  - AEMET clima diario 2017 (estación más cercana) + imputación + agregación anual
  - Revisar calidad de datos (missing / tipos / outliers)
  - Analizar target (`cultivo_grupo`) y sesgos geográficos (provincia)
  - PSI por provincia vs global
  - Correlaciones numéricas

**Artefactos**
- `outputs/eda/dataset_final_2017_full.csv`
- `outputs/eda/model/dataset_final_2017_model.csv` (sin clase "Otros")
- `outputs/eda/analysis/`
- Checks intermedios en `outputs/eda/` y cache en `cache/`

"""

import os
import re
import json
import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from tqdm import tqdm
import time, random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from http.client import RemoteDisconnected
from json import JSONDecodeError
from datetime import datetime

# =========================
# Configuración y rutas
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJtYXJnYV9tZ18xM0Bob3RtYWlsLmNvbSIsImp0aSI6IjAxZDRkZDU2LTMzYzktNGQ5ZS04MmYyLTc2NWJlMzc1YjRiZSIsImlzcyI6IkFFTUVUIiwiaWF0IjoxNzU3MzI5ODAyLCJ1c2VySWQiOiIwMWQ0ZGQ1Ni0zM2M5LTRkOWUtODJmMi03NjViZTM3NWI0YmUiLCJyb2xlIjoiIn0.rhgjNsswbnMadh8S4dnKtU9dJKz5MQ7nnPUa51G4z4A"  # <-- pon tu token AEMET
DATOS_DIR = Path("data")
OUTPUT_DIR = Path("outputs/eda")
CACHE_DIR = Path("cache")
for p in (OUTPUT_DIR, CACHE_DIR):
    p.mkdir(parents=True, exist_ok=True)

CLIMATE_2017_STATION_CSV = OUTPUT_DIR / "clima_2017_por_estacion.csv"
CLIMATE_2017_STATION_IMPUTED_CSV = OUTPUT_DIR / "clima_2017_por_estacion_imputado.csv"
CLIMATE_2017_PROFILE_CSV = OUTPUT_DIR / "clima_2017_por_perfil.csv"
CLIMATE_2017_PROFILE_IMPUTED_CSV = OUTPUT_DIR / "clima_2017_por_perfil_imputado.csv"

# =========================
# Utilidades
# =========================
def _coerce_decimal(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)): return float(x)
    s = str(x).strip().replace(",", ".").replace("_", ".")
    try: return float(s)
    except: return np.nan

def _last_day_of_month(year: int, month: int) -> int:
    import calendar
    return calendar.monthrange(year, month)[1]

# ---------- Lectura robusta PANGAEA ----------
HDR_KEYWORDS = ["latitude", "longitude", "lcc", "description", "corine", "sample id", "depth", "horizon"]
def _detect_header_idx(path: Path) -> int:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    best_idx, best_tabs, best_score = None, -1, -1
    for i, raw in enumerate(lines[:1000]):
        if raw.startswith("/"):  # comentarios PANGAEA
            continue
        tabs = raw.count("\t")
        if tabs < 5:
            continue
        low = raw.lower()
        score = sum(kw in low for kw in HDR_KEYWORDS)
        if score >= 3 and (tabs > best_tabs or (tabs == best_tabs and score > best_score)):
            best_idx, best_tabs, best_score = i, tabs, score

    if best_idx is not None:
        logging.info(f"Cabecera detectada en línea {best_idx} (tabs={best_tabs}, score={best_score}) en {path.name}")
        return best_idx

    # Fallback: mayor nº de tabs
    for i, raw in enumerate(lines[:1000]):
        if raw.startswith("/"): continue
        tabs = raw.count("\t")
        if tabs > best_tabs:
            best_idx, best_tabs = i, tabs
    if best_idx is not None:
        logging.warning(f"Cabecera por fallback (máx tabs) en línea {best_idx} para {path.name}")
        return best_idx

    logging.warning(f"No se pudo detectar cabecera fiable en {path.name}; uso línea 0")
    return 0

def _read_pangaea_tab(path: Path) -> pd.DataFrame:
    hdr = _detect_header_idx(path)
    df = pd.read_csv(
        path, sep="\t", header=hdr, engine="python",
        on_bad_lines="skip", dtype=str
    )
    # elimina columnas 'Unnamed'
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed", na=False)].copy()
    # quita primera fila si parecen unidades
    if len(df) > 0:
        sample = " ".join(map(str, df.iloc[0, :15].tolist()))
        if any(sym in sample for sym in ("[", "]", "°", "‰")):
            df = df.iloc[1:].reset_index(drop=True)
    return df

# ---------- Resolver IDs y coords ----------
def _resolve_ids(perfiles: pd.DataFrame, horizontes: pd.DataFrame):
    pf = perfiles.copy()
    hz = horizontes.copy()

    # profile_id en perfiles = Sample ID (Unique identification number ...) SIN sufijo ".1"
    pf_cols = [c for c in pf.columns if c.lower().startswith("sample id (unique identification number") and not c.endswith(".1")]
    if pf_cols:
        pf = pf.rename(columns={pf_cols[0]: "profile_id"})
    elif "profile_id" not in pf.columns:
        pf["profile_id"] = np.arange(len(pf)).astype(str)
        logging.warning("No se encontró columna de Profile ID en 'profile'; se creó sintética.")

    # profile_id en horizontes = mismo nombre PERO con sufijo ".1"
    hz_cols = [c for c in hz.columns if c.lower().startswith("sample id (unique identification number") and c.endswith(".1")]
    if hz_cols:
        hz = hz.rename(columns={hz_cols[0]: "profile_id"})
    elif "profile_id" not in hz.columns:
        logging.warning("No se encontró Profile ID en 'horizons'; joins pueden ser incompletos.")

    # horizon_id opcional
    hz_hid = [c for c in hz.columns if c.lower().startswith("sample id (unique identification number") and not c.endswith(".1")]
    if hz_hid and "horizon_id" not in hz.columns:
        hz = hz.rename(columns={hz_hid[0]: "horizon_id"})

    # lat/lon -> 'lat','lon'
    def _push_latlon(df):
        for name in list(df.columns):
            low = name.lower()
            if low == "latitude":
                df["lat"] = df[name].map(_coerce_decimal)
            if low == "longitude":
                df["lon"] = df[name].map(_coerce_decimal)
            if name == "Latitude": df["lat"] = df[name].map(_coerce_decimal)
            if name == "Longitude": df["lon"] = df[name].map(_coerce_decimal)
        return df
    pf = _push_latlon(pf)
    hz = _push_latlon(hz)

    if "profile_id" in pf.columns: pf["profile_id"] = pf["profile_id"].astype(str)
    if "profile_id" in hz.columns: hz["profile_id"] = hz["profile_id"].astype(str)
    return pf, hz

# ---------- Variable objetivo: cultivo desde Description ----------
def _ensure_cultivo_from_description(perfiles: pd.DataFrame) -> pd.DataFrame:
    pf = perfiles.copy()
    # Candidatas que contengan 'description'
    cand = [c for c in pf.columns if "description" in c.lower()]
    pref = [c for c in cand if "vegetation" in c.lower() and "provided" in c.lower()]
    col = pref[0] if pref else (cand[0] if cand else None)
    if not col:
        raise KeyError("No se encontró la columna de 'Description' en 'profile'. Revisa la cabecera detectada.")
    pf["cultivo"] = pf[col].astype(str).str.strip()
    # Exporta categorías únicas
    cats = (pf["cultivo"].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique())
    cats_sorted = np.sort(cats)
    pd.Series(cats_sorted, name="categoria_cultivo").to_csv(OUTPUT_DIR / "cultivo_categorias_unicas.csv", index=False)
    logging.info(f"Columna 'cultivo' creada a partir de: {col}. Categorías únicas: {len(cats_sorted)} -> cultivo_categorias_unicas.csv")
    return pf


# --- helpers ---
import unicodedata, re

def normalize_text(x: str) -> str:
    if not isinstance(x, str): return ""
    s = x.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def assign_cultivo_group(value: str) -> str:
    """
    Agrupa cultivos/uso del suelo en categorías robustas para un clasificador.
    Mantiene NaN/vacío en 'Otros'.
    Categorías clave: Forest(al), Matorral, Pastizal, Cereal de invierno, Maíz,
    Arrozal, Hortaliza, Viñedo, Olivar, Frutal, Frutos secos, Cítricos,
    Herbáceos industriales, Agua, Desnudo, Mixto, Urbano, Industrial, Barbecho, Otros.
    """
    if not isinstance(value, str):
        return "Otros"
    s = normalize_text(value)
    if s == "" or s in {"na","nan","none","null"}:
        return "Otros"

    # ===== Patrones CORINE / mixtos =====
    if re.search(r"\b(non[- ]?irrigated\s*arable\s*land|secano(\s*no\s*regado)?|tierras?\s*de\s*labr[ao]\s*no\s*regad)\b", s):
        return "Cereal de invierno"
    if re.search(r"\b(permanently\s*irrigated\s*land|regadi[oó]|tierras?\s*de\s*regadi[oó])\b", s):
        return "Hortaliza"
    if re.search(r"\b(complex\s*cultivation\s*patterns|patrones?\s*complejos?\s*de\s*cultivo|mosaico\s*de\s*cultivos?)\b", s):
        return "Mixto"
    if re.search(r"\b(annual\s*crops\s*associated\s*with\s*permanent\s*crops|cultivos?\s*anuales?\s*asociados?\s*con\s*le[ñn]osos?)\b", s):
        return "Mixto"

    # ===== Herbáceos INDUSTRIALES (nueva categoría agregada) =====
    if re.search(r"\b(tabaco|tobacco|nicotiana|algod[oó]n|cotton|gossypium|remolacha(\s+azucarera)?|sugar\s*beet|colza|rapeseed|canola|brassica\s+napus|ca[ñn]a\s*de\s*az[uú]car|sugar\s*cane|saccharum)\b", s):
        return "Herbáceos industriales"

    # ===== Herbáceos alimentarios principales =====
    if re.search(r"\b(cereal|arable|crop|cultivo|trigo|wheat|cebada|barley|avena|oat|centeno|rye|triticum|hordeum|secale|rastrojo|stubble)\b", s):
        return "Cereal de invierno"
    if re.search(r"\b(ma[ií]z|maizal|corn|zea\s+mays)\b", s):
        return "Maíz"
    if re.search(r"\b(rice|arroz|oryza|paddy)\b", s):
        return "Arrozal"
    # Hortaliza / huerta (incluye "vegetales" y plurales)
    if re.search(r"\b(vegetable(s)?|vegetal(?:es)?|hortali\w*|huerta)\b", s) or \
    re.search(r"\b(tomat\w*|pepino\w*|cucumber\w*|cebolla\w*|ajo\w*|onion\w*|garlic\w*|lettuce\w*|lechuga\w*|melon\w*|sandia\w*|pepper\w*|pimiento\w*|berenjen\w*|eggplant\w*|calabacin\w*|zucchini\w*|invernadero|greenhouse|plasticulture|tunnel\s*greenhouse)\b", s):
        return "Hortaliza"
    if re.search(r"\b(sunflower|girasol|helianthus)\b", s):
        return "Girasol"

    # ===== Leñosos =====
    if re.search(r"\b(vine(yard)?|vitis|grape|vinedo|vi[ñn]a(s)?|vinal(es)?)\b", s):
        return "Viñedo"
    if re.search(r"\b(olive|olivar|olivo|olea)\b", s):
        return "Olivar"
    if re.search(r"\b(citrus|naranj|orange|lim[oó]n|lemon|mandar|pomelo|grapefruit)\b", s):
        return "Cítricos"
    # Frutos secos (almendro/nogal/avellano/pistacho)
    if re.search(r"\b(almendr\w+|almonds?)\b", s) or \
    re.search(r"\b(avellan\w+|hazel\w*|nogal(?:es)?|walnuts?)\b", s) or \
    re.search(r"\b(pistach\w+|corylus|juglans|pistacia|prunus\s+dulcis)\b", s):
        return "Frutos secos"
    # Frutal general (incluye castaño para no fragmentar clases)
    if re.search(r"\b(castanea\s+sativa|casta[ñn]o|higuera|fig|ficus\s*carica|granado|pomegranate|manzan|apple|pera|pear|melocot|peach|cerezo|cherry|ciruelo|plum|albaricoque|apricot|kiwi|actinidia|mango|aguacate|avocado|orchard|fruit\s*trees?)\b", s):
        return "Frutal"
    if re.search(r"\b(palmeral|date\s*palm|phoenix\s+dactylifera)\b", s):
        return "Frutal"

    # ===== Naturales =====
    # Pastizal
    if re.search(r"\b(pastures?|pasture|pradera|prado|pasto|meadow|heno|hay|herbazal|stipa(\s|$)|poaceae(\s|$)|festuca|nardus|bromion|bromus)\b", s):
        return "Pastizal"
    # Matorral
    if re.search(r"\b(scrub|matorral|maquis|garrigue|heath|broom|monte\s*bajo|maleza|xeri[cx]\w*\s*vegetation|jaral|brezal)\b", s):
        return "Matorral"
    if re.search(r"\b(cistus|cystus|erica|genista|retama|rosmarinus|thymus|halimium|pistacia|ulex|juniper(us)?)\b", s):
        return "Matorral"
    # ===== Forestal subdividido =====

    # Coníferas (pinos, abetos, cedros, etc.)
    if re.search(r"\b(pin(us)?|pinar|abies|abeto|cedrus|larix|picea|pinsapo|pine\s*reforestation|reforestaci[oó]n\s*de?\s*pino)\b", s):
        return "Forest-Coníferas"

    # Frondosas autóctonas (encina, roble, haya, olmo, fresno, chopos/álamos)
    if re.search(r"\b(quercus|encina|roble|fagus|haya|ulmus|olmo|fraxinus|fresno|populus|chopo|álamo|betula|alnus|acer)\b", s):
        return "Forest-Frondosas"

    # Eucalipto (monte productivo diferenciado)
    if re.search(r"\b(eucalyptus|eucalipt(al)?)\b", s):
        return "Forest-Eucalipto"

    # Forestal genérico (si no ha caído antes)
    if re.search(r"\b(forest|bosque|woodland|robledal|quejigal|dehesa|montado|broadleaved\s*trees?)\b", s):
        return "Forest-Otros"

    # ===== Agua / humedal =====
    if re.search(r"\b(water|lake|river|embalse|wetland|marsh|lagoon|estuari|humedal|peat\s*bog|bog|salt\s*marsh|salinas|intertidal|reedbed|carrizal|juncus|carex|sphagnum|halophila)\b", s):
        return "Agua"

    # ===== Desnudo =====
    if re.search(r"\b(few\s*cover|few\s*vegetation|sparse(ly)?\s*vegetated|bare\s*soil|bare|rock|roca|sand|arena|gravel|grava|duna|dune|playa|beach|burnt\s*areas?)\b", s):
        return "Desnudo"

    # ===== Otros usos =====
    if re.search(r"\b(fallow|barbecho|abandoned)\b", s):
        return "Barbecho"
    if re.search(r"\b(urban|urbano|residential|ciudad|pueblo|edific)\b", s):
        return "Urbano"
    if re.search(r"\b(industry|industrial|factory|poligono|mineral\s*extraction|quarry|cantera|dump\s*site|landfill|construction\s*site)\b", s):
        return "Industrial"
    if "mixed" in s or "mixto" in s or "mosaic" in s:
        return "Mixto"
    # Humedal genérico (más familias)
    if re.search(r"\b(scirpus|schoenus|typha|phragmites)\b", s):  # juncáceas/eneas/carrizo
        return "Agua"

    # Comunidades botánicas alpinas → Pastizal (muy típico en tus 'Otros')
    if re.search(r"\b(festucion|nardion|bromion|poion|seslerion)\b", s):
        return "Pastizal"

    # Comunidades psamófilas/dunas → Desnudo
    if re.search(r"\b(ammophil|elymion|dune\s*grass|psammo\w+)\b", s):
        return "Desnudo"

    # Frondosas de ribera → Forest-Frondosas
    if re.search(r"\b(salix|populus\s*nigra|fraxinus\s*angustifolia)\b", s):
        return "Forest-Frondosas"
    
    # Herbáceos industriales (captura más variantes)
    if re.search(r"\b(tabaco|tobacco|nicotiana|tabacco|tabac+o|tabaco\s*plantation)\b", s):
        return "Herbáceos industriales"

    # Frutal (añadimos Castanea)
    if re.search(r"\b(castanea\s+sativa|casta[ñn]o)\b", s):
        return "Frutal"

    # Pastizal (más gramíneas alpinas)
    if re.search(r"\b(festuca|nardus|bromus|bromion|nardion|festucion)\b", s):
        return "Pastizal"

    # Agua (plantas acuáticas y humedales)
    if re.search(r"\b(juncus|carex|sphagnum|scirpus|schoenus|typha|phragmites)\b", s):
        return "Agua"

    # Desnudo (dunas y psamófitas)
    if re.search(r"\b(dunes?|ammophil|elymion|psammo\w+)\b", s):
        return "Desnudo"

    # Helechos
    if re.search(r"\b(pteridium|helecho)\b", s):
        return "Matorral"

    # Vegetación ribera
    if re.search(r"\b(riparian\s*vegetation)\b", s):
        return "Forest-Frondosas"

    # Vegetación halófila / salinas
    if re.search(r"\b(salicornia|suaeda|atriplex|arthrocnemum|halogeton|limonium|frankenia)\b", s):
        return "Agua"

    # Vegetación nitrófila / gipsícola / termófila
    if re.search(r"\b(nitrophil\w*|gypsicol\w*|termophil\w*)\b", s):
        return "Matorral"

    # Especies arbóreas dispersas
    if re.search(r"\b(buxus|tilia|platanus|robinia)\b", s):
        return "Forest-Frondosas"
    if re.search(r"\b(juglans|junglans\s*regia|nogal)\b", s):
        return "Frutos secos"
    if re.search(r"\b(eucalypth?us)\b", s):   # captura "Eucalypthus" mal escrito
        return "Forest-Eucalipto"

    # Arbustos mediterráneos
    if re.search(r"\b(spartium|calluna|clematis|launaea|thymelaea|artemisia|helianthemum)\b", s):
        return "Matorral"

    # Gramíneas
    if re.search(r"\b(cynodon|brachypodium)\b", s):
        return "Pastizal"

    # Pinares
    if re.search(r"\b(talled\s*pine\s*area)\b", s):
        return "Forest-Coníferas"

    # Lamiaceae / Laminaceae con contexto
    if re.search(r"\b(lami?naceae)\b", s):  # captura 'lamiaceae' y 'laminaceae'
        # Arbustivo / matorral aromático
        if re.search(r"\b(heath|shrub|scrub|matorr|maquis|garrigue|jaral|brezal|thymus|rosmarinus|lavandula|cistus)\b", s):
            return "Matorral"
        # Herbazal / pradera
        if re.search(r"\b(past|meadow|pradera|prado|herbazal|grass|poa|festuca|bromus|stipa)\b", s):
            return "Pastizal"
        # Sin contexto claro: preferimos Pastizal para el modelo
        return "Pastizal"

    # Plantago -> Pastizal (si no se clasificó ya)
    if re.search(r"\b(plantago)\b", s):
        return "Pastizal"

    # Artemisia/Helianthemum -> Matorral (si no se clasificó ya)
    if re.search(r"\b(artemisia|helianthemum)\b", s):
        return "Matorral"

    # ======= CATCH-ALL para que "Otros" solo tenga NaN/vacíos =======
    # Si llegamos aquí, la descripción no era NaN y no ha matcheado nada específico.

    # 1) Señales de comunidades/vegetación genérica
    if re.search(r"\b(riparian\s*vegetation)\b", s):
        return "Forest-Frondosas"
    if re.search(r"\b(nitrophil\w*|gypsicol\w*|termophil\w*)\b", s):
        return "Matorral"
    if re.search(r"\b(vegetation|community|association|alliance)\b", s):
        # genérico: sin señal de agua → Matorral por defecto
        if re.search(r"\b(wet|reed|marsh|bog|salt|salin|halophil|humed|carriz|junc|carex|sphagn|typha|phragmites|lagoon|estuari|intertidal)\b", s):
            return "Agua"
        if re.search(r"\b(grass|meadow|past|gramin|poa|festuca|stipa|bromus|pradera|prado|pasto|herbazal)\b", s):
            return "Pastizal"
        if re.search(r"\b(shrub|scrub|matorr|maquis|garrigue|heath|broom|jaral|brezal|maleza|monte\s*bajo)\b", s):
            return "Matorral"
        if re.search(r"\b(tree|arbore|forest|bosque|woodland|frondos|broadleaved)\b", s):
            return "Forest-Frondosas"
        return "Matorral"

    # 2) Botánica: familias/géneros no capturados arriba
    #   - Halófilas → Agua
    if re.search(r"\b(salicornia|suaeda|atriplex|arthrocnemum|halogeton|limonium|frankenia)\b", s):
        return "Agua"
    #   - Gramíneas / praderas → Pastizal
    if re.search(r"\b(poa|festuca|bromus|stipa|cynodon|brachypodium|poaceae)\b", s):
        return "Pastizal"
    #   - Arbustos mediterráneos → Matorral
    if re.search(r"\b(cistus|erica|genista|retama|rosmarinus|thymus|helianthemum|launaea|calluna|clematis|artemisia|spartium)\b", s):
        return "Matorral"
    #   - Frondosas de ribera / parque → Forest-Frondosas
    if re.search(r"\b(salix|populus\s*nigra|platanus|tilia|ulmus|robinia|fraxinus)\b", s):
        return "Forest-Frondosas"
    #   - Coníferas / pinares residuales → Forest-Coníferas
    if re.search(r"\b(pinus|pino|pinar|picea|larix|cedrus|abies|pinsapo)\b", s):
        return "Forest-Coníferas"
    #   - Eucalipto → Forest-Eucalipto
    if re.search(r"\b(eucalyptus|eucalipt)\b", s):
        return "Forest-Eucalipto"
    #   - Frutales sueltos (incluye castaño, prunus genérico) → Frutal/Frutos secos
    if re.search(r"\b(castanea\s+sativa|casta[ñn]o|prunus\s+|orchard|fruit\s*trees?)\b", s):
        return "Frutal"
    if re.search(r"\b(juglans|walnut|nogal|corylus|hazel|pistacia|pistach)\b", s):
        return "Frutos secos"

    # 3) Medio físico
    #   - Dunas / playas / arena → Desnudo
    if re.search(r"\b(dunes?|ammophil|elymion|psammo\w+|beach|playa|sand|arena)\b", s):
        return "Desnudo"
    #   - Plantations genéricas (sin especificar cultivo) → Forest-Otros (suelen ser leñosos)
    if re.search(r"\b(plantation|reforestaci[oó]n)\b", s):
        # si no era pino/eucalipto (capturado antes), lo dejamos en forestal genérico
        return "Forest-Otros"
    #   - Campos/tierras cultivadas genéricas → Cereal de invierno por defecto
    if re.search(r"\b(arable|crop|cultivo|campo)\b", s):
        return "Cereal de invierno"
    #   - Huerta genérica → Hortaliza
    if re.search(r"\b(vegetable(s)?|vegetal(?:es)?|hortali\w*|huerta)\b", s):
        return "Hortaliza"

    return "Otros"


# -------------------------
# Parser robusto DMS AEMET
# -------------------------
_DNUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

def _parse_aemet_dms(s: str) -> float:
    if pd.isna(s): 
        return np.nan
    t = str(s).strip().upper()
    if not t: 
        return np.nan

    # Hemisferio
    hem = None
    if t and t[-1] in "NSEW":
        hem = t[-1]
        t = t[:-1].strip()

    # Normaliza símbolos
    t_norm = (t.replace("º", " ")
                .replace("°", " ")
                .replace("’", " ")
                .replace("'", " ")
                .replace("″", " ")
                .replace('"', " ")
                .replace("  ", " ").strip())

    # 0) ¿Formato compacto? (¡comprobar ANTES que el caso de D/M/S con separadores!)
    #   - Solo dígitos (sin espacios/separadores)
    only_digits = re.sub(r"\D", "", t)
    if only_digits and t_norm.replace(" ", "") == only_digits:
        L = len(only_digits)
        # DDMM / DDMMSS  (lat)   y   DDDMM / DDDMMSS (lon)
        if L >= 4:
            if L >= 6:  # ...SS
                sec = float(only_digits[-2:])
                minu = float(only_digits[-4:-2])
                deg  = float(only_digits[:-4])
            else:       # ...MM
                sec = 0.0
                minu = float(only_digits[-2:])
                deg  = float(only_digits[:-2])
            val = deg + minu/60.0 + sec/3600.0
            if hem in ("S", "W"): 
                val = -abs(val)
            elif hem in ("N", "E"):
                val =  abs(val)
            # Validación de rango
            if -180 <= val <= 180:
                return val
            return np.nan

    # 1) ¿Decimal puro?
    try:
        val = float(t_norm.replace(",", "."))
        if hem in ("S", "W"): 
            val = -abs(val)
        elif hem in ("N", "E"):
            val =  abs(val)
        if -180 <= val <= 180:
            return val
        return np.nan
    except Exception:
        pass

    # 2) D, D M, D M S con separadores
    nums = _DNUM.findall(t_norm)
    if len(nums) >= 1:
        deg = float(nums[0].replace(",", "."))
        minu = float(nums[1].replace(",", ".")) if len(nums) >= 2 else 0.0
        sec  = float(nums[2].replace(",", ".")) if len(nums) >= 3 else 0.0
        val = deg + minu/60.0 + sec/3600.0
        if hem in ("S", "W"): 
            val = -abs(val)
        elif hem in ("N", "E"):
            val =  abs(val)
        if -180 <= val <= 180:
            return val
        return np.nan

    return np.nan

# -------------------------
# Descarga + parseo estaciones
# -------------------------
_SESSION = None
_SESSION_USES = 0
def _new_session():
    s = requests.Session()
    retry = Retry(
        total=8, connect=8, read=8, backoff_factor=0.7,
        status_forcelist=[429,500,502,503,504],
        allowed_methods=["GET"], raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=40)
    s.mount("https://", adapter); s.mount("http://", adapter)
    s.headers.update({"Accept":"application/json", "Connection":"keep-alive"})
    return s

def _get_session():
    global _SESSION, _SESSION_USES
    if _SESSION is None or _SESSION_USES >= 25:   # rota cada 25 requests
        _SESSION = _new_session()
        _SESSION_USES = 0
    _SESSION_USES += 1
    return _SESSION

def _sleep_jitter(base=0.8, spread=0.8):  # 0.8–1.6 s
    time.sleep(base + random.random()*spread)

def _short(url, n=100): 
    return url if len(url)<=n else url[:n]+"…"

def _get_json(url, headers, timeout=40, tries=5):
    last = None
    for i in range(1, tries+1):
        try:
            r = _get_session().get(url, headers=headers, timeout=timeout, stream=False)
            if r.status_code == 204 or not r.content:
                return None
            r.raise_for_status()
            try:
                return r.json()
            except JSONDecodeError as je:
                last = je
                logging.warning(f"JSONDecode en {_short(url)} (intento {i}/{tries})")
        except (requests.exceptions.ConnectionError, RemoteDisconnected) as ce:
            last = ce
            logging.warning(f"Conexión abortada {_short(url)} (intento {i}/{tries}): {ce}")
        except requests.HTTPError as he:
            last = he
            sc = r.status_code
            logging.warning(f"HTTP {sc} en {_short(url)} (intento {i}/{tries})")
            if 400 <= sc < 500 and sc != 429:
                break
        time.sleep(min(0.6*(2**(i-1)), 5.0) + random.random()*0.5)
    raise RuntimeError(f"Falló GET tras {tries} intentos: {_short(url)} -> {last}")

def get_aemet_stations() -> pd.DataFrame:
    cache_file = CACHE_DIR / "stations.json"

    def _load():
        url = "https://opendata.aemet.es/opendata/api/valores/climatologicos/inventarioestaciones/todasestaciones"
        headers = {"api_key": API_KEY}
        meta = _get_json(url, headers=headers, timeout=45, tries=5)
        payload = _get_json(meta["datos"], headers=headers, timeout=60, tries=6)
        df = pd.DataFrame(payload)
        df.to_json(cache_file, orient="records")
        return df

    df = (pd.read_json(cache_file, orient="records") if cache_file.exists() else _load())

    # Parse coords
    df["lat"] = df["latitud"].apply(_parse_aemet_dms)
    df["lon"] = df["longitud"].apply(_parse_aemet_dms)
    df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)].copy()

    # Parse vigencia
    for c in ("fechaAlta", "fechaBaja"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")
        else:
            df[c] = pd.NaT

    # Activas en 2017 (intersección de periodos)
    y0 = pd.Timestamp("2017-01-01")
    y1 = pd.Timestamp("2017-12-31 23:59:59")
    alta_ok = (df["fechaAlta"].isna()) | (df["fechaAlta"] <= y1)
    baja_ok = (df["fechaBaja"].isna()) | (df["fechaBaja"] >= y0)
    df = df[alta_ok & baja_ok].copy()

    valid = df[["lat","lon"]].notna().all(axis=1).sum()
    logging.info(f"Inventario: {len(df)} estaciones activas en 2017; con coords válidas: {valid}")

    if valid == 0:
        # fuerza recarga si algo raro
        df = _load()
        df["lat"] = df["latitud"].apply(_parse_aemet_dms)
        df["lon"] = df["longitud"].apply(_parse_aemet_dms)
        df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)].copy()
        for c in ("fechaAlta","fechaBaja"):
            df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")
        df = df[((df["fechaAlta"].isna()) | (df["fechaAlta"] <= y1)) &
                ((df["fechaBaja"].isna()) | (df["fechaBaja"] >= y0))].copy()
        if df[["lat","lon"]].notna().all(axis=1).sum() == 0:
            raise RuntimeError("Inventario AEMET sin coords/vigencias válidas para 2017.")

    return df.reset_index(drop=True)
# -------------------------
# Asignación Haversine
# -------------------------
def _ensure_datetime(s):
    if not pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce", format="mixed")
    return s

def _active_mask_for_year(stations: pd.DataFrame, year: int) -> pd.Series:
    """Devuelve máscara booleana de estaciones activas en cualquier momento del año 'year'."""
    st = stations.copy()
    # Asegurar fechas como datetime
    for c in ("fechaAlta","fechaBaja"):
        if c not in st.columns:
            st[c] = pd.NaT
    st["fechaAlta"] = _ensure_datetime(st["fechaAlta"])
    st["fechaBaja"] = _ensure_datetime(st["fechaBaja"])

    y0 = pd.Timestamp(f"{year}-01-01")
    y1 = pd.Timestamp(f"{year}-12-31 23:59:59")

    alta_ok = st["fechaAlta"].isna() | (st["fechaAlta"] <= y1)
    baja_ok = st["fechaBaja"].isna() | (st["fechaBaja"] >= y0)
    return (alta_ok & baja_ok)

def assign_nearest_station(perfiles: pd.DataFrame, stations: pd.DataFrame, year: int = 2017) -> pd.DataFrame:
    """
    Asigna SIEMPRE estación:
      - Prioriza estaciones ACTIVAS en 'year'.
      - Si la más cercana no está vigente, usa la siguiente más cercana que sí lo esté.
      - Si ninguna está vigente, asigna la más cercana igualmente y marca 'station_active_2017=False'.
    Guarda check CSV en OUTPUT_DIR/perfiles_estaciones_check.csv
    """
    pf = perfiles.copy()

    # Asegurar lat/lon en perfiles
    if "lat" not in pf.columns or "lon" not in pf.columns:
        if "Latitude" in pf.columns: pf["lat"] = pf["Latitude"].map(_coerce_decimal)
        if "Longitude" in pf.columns: pf["lon"] = pf["Longitude"].map(_coerce_decimal)
    if not {"lat","lon"}.issubset(pf.columns):
        raise ValueError("No se encontraron columnas lat/lon en 'profile' para asignar estación.")

    pf["lat"] = pd.to_numeric(pf["lat"], errors="coerce")
    pf["lon"] = pd.to_numeric(pf["lon"], errors="coerce")
    pfv = pf.dropna(subset=["lat","lon"]).copy()
    if pfv.empty:
        raise ValueError("Ningún perfil tiene lat/lon válidos.")

    # Asegurar estaciones válidas y máscara de vigencia
    st = stations.dropna(subset=["lat","lon"]).copy()
    if st.empty:
        raise ValueError("No hay estaciones con lat/lon válidos.")
    active_mask = _active_mask_for_year(st, year).to_numpy()

    # Distancias (Haversine) perfil↔todas las estaciones
    coords = pfv[["lat","lon"]].to_numpy(dtype=float)
    ST = st[["lat","lon"]].to_numpy(dtype=float)
    pf_lat = np.deg2rad(coords[:,0])[:,None]
    pf_lon = np.deg2rad(coords[:,1])[:,None]
    st_lat = np.deg2rad(ST[:,0])[None,:]
    st_lon = np.deg2rad(ST[:,1])[None,:]
    dphi = st_lat - pf_lat
    dl   = st_lon - pf_lon
    a = np.sin(dphi/2.0)**2 + np.cos(pf_lat)*np.cos(st_lat)*np.sin(dl/2.0)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1.0-a))
    dist_km = 6371.0 * c

    # Orden de estaciones por cercanía para cada perfil
    order = np.argsort(dist_km, axis=1)

    # Elegir la primera activa; si ninguna, la más cercana
    chosen_idx = np.empty(order.shape[0], dtype=int)
    chosen_dist = np.empty(order.shape[0], dtype=float)
    chosen_rank = np.empty(order.shape[0], dtype=int)
    chosen_active = np.empty(order.shape[0], dtype=bool)

    for i in range(order.shape[0]):
        ord_i = order[i]
        # máscara de activas siguiendo el orden por distancia
        active_ord = active_mask[ord_i]
        if active_ord.any():
            k = int(np.argmax(active_ord))  # primera True en ese orden
            j = ord_i[k]
            chosen_idx[i] = j
            chosen_dist[i] = float(dist_km[i, j])
            chosen_rank[i] = k + 1           # 1 = la más cercana, 2 = segunda, ...
            chosen_active[i] = True
        else:
            # fallback: la más cercana (aunque no activa)
            j = ord_i[0]
            chosen_idx[i] = j
            chosen_dist[i] = float(dist_km[i, j])
            chosen_rank[i] = 1
            chosen_active[i] = False

    st_ix = st.reset_index(drop=True).iloc[chosen_idx]

    out = pfv[["profile_id"]].copy()
    out["nearest_station"]        = st_ix["indicativo"].astype(str).values
    out["station_name"]           = st_ix.get("nombre", pd.Series([""]*len(out))).astype(str).values
    out["station_provincia"]      = st_ix.get("provincia", pd.Series([""]*len(out))).astype(str).values
    out["distance_km"]            = chosen_dist
    out["station_active_2017"]    = chosen_active
    out["station_rank_used"]      = chosen_rank  # 1=primera más cercana, 2=segunda, etc.

    # (opcional) avisos de salud
    n_inactive = (~out["station_active_2017"]).sum()
    if n_inactive:
        logging.warning(f"{n_inactive} perfiles no encontraron estación activa en {year}; se asignó la más cercana inactiva.")

    out.to_csv(OUTPUT_DIR / "perfiles_estaciones_check.csv", index=False)
    return out

# =========================
# Clima 2017  (descarga + imputación)
# =========================
def _clean_daily_payload(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", format="mixed")
        df["month"] = df["fecha"].dt.month
    numeric_cols = [c for c in df.columns if c not in ("indicativo","nombre","provincia","fecha","month")]
    for col in numeric_cols:
        s = df[col].astype(str).str.strip()
        if col.lower().startswith("prec"): s = s.replace({"Ip":"0","ip":"0"})
        s = s.str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(s, errors="coerce")
    return df

def _range_url(station_id: str, start: str, end: str) -> str:
    return ("https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/"
            f"datos/fechaini/{start}T00:00:00UTC/fechafin/{end}T23:59:59UTC/estacion/{station_id}")

def _fetch_range(stid: str, start: str, end: str, headers: dict, meta_tries=3) -> pd.DataFrame:
    """Intenta meta->datos con 'meta refresh' si la short URL muere."""
    meta_url = _range_url(stid, start, end)
    meta = _get_json(meta_url, headers=headers, timeout=45, tries=5)
    if not meta or "datos" not in meta or not meta.get("datos"):
        logging.info(f"[{stid} {start}..{end}] sin datos (meta vacío/204).")
        return pd.DataFrame()
    short = meta["datos"]
    for k in range(meta_tries):
        try:
            payload = _get_json(short, headers=headers, timeout=60, tries=6)
            if payload:
                df = _clean_daily_payload(pd.DataFrame(payload))
                return df
            logging.info(f"[{stid} {start}..{end}] payload vacío (k={k+1}).")
        except Exception as e:
            logging.warning(f"[{stid} {start}..{end}] short URL falló (k={k+1}): {e}")
            meta = _get_json(meta_url, headers=headers, timeout=45, tries=5)  # refresh
            short = meta.get("datos")
        time.sleep(0.4 + random.random()*0.5)
    return pd.DataFrame()

def get_daily_climate_year(stid: str, year=2017) -> pd.DataFrame:
    cache_file = CACHE_DIR / f"climate_{stid}_{year}.json"
    headers = {"api_key": API_KEY}

    if cache_file.exists():
        try:
            df = pd.read_json(cache_file, orient="records")
            return _clean_daily_payload(df)
        except Exception as e:
            logging.warning(f"Cache corrupto {cache_file.name}: {e}")

    # estrategia descendente: año -> semestres -> trimestres -> meses
    ranges = [
        [(f"{year}-01-01", f"{year}-12-31")],
        [(f"{year}-01-01", f"{year}-06-30"), (f"{year}-07-01", f"{year}-12-31")],
        [(f"{year}-01-01", f"{year}-03-31"), (f"{year}-04-01", f"{year}-06-30"),
         (f"{year}-07-01", f"{year}-09-30"), (f"{year}-10-01", f"{year}-12-31")],
        [(f"{year}-{m:02d}-01", f"{year}-{m:02d}-{_last_day_of_month(year,m):02d}") for m in range(1,13)],
    ]

    got_any = False
    frames = []
    for level, rr in enumerate(ranges, start=1):
        frames = []
        got_any = False
        for (start, end) in rr:
            try:
                df = _fetch_range(stid, start, end, headers=headers, meta_tries=3)
            except Exception as e:
                logging.warning(f"[{stid} {start}..{end}] error: {e}")
                df = pd.DataFrame()
            if not df.empty:
                frames.append(df); got_any = True
            _sleep_jitter(0.3, 0.5)  # descanso entre subrangos
        if got_any:
            break  # suficiente en este nivel

    if not got_any:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    if not out.empty:
        out.to_json(cache_file, orient="records")
    return out

# checkpoint para reanudar
DONE_FILE = CACHE_DIR / "climate_2017_done.txt"
FAIL_FILE = CACHE_DIR / "climate_2017_failed.txt"

def _load_done():
    if DONE_FILE.exists():
        return set(x.strip() for x in DONE_FILE.read_text(encoding="utf-8").splitlines() if x.strip())
    return set()

def _mark_done(stid):
    with open(DONE_FILE, "a", encoding="utf-8") as f: f.write(stid + "\n")

def _mark_fail(stid):
    with open(FAIL_FILE, "a", encoding="utf-8") as f: f.write(stid + "\n")

def build_and_save_climate_2017(perfiles: pd.DataFrame):
    valid = perfiles.dropna(subset=["nearest_station"]).copy()
    unique_stations = list(pd.unique(valid["nearest_station"].astype(str)))
    done = _load_done()

    station_frames = []
    for stid in tqdm(unique_stations, desc="Estaciones 2017", total=len(unique_stations)):
        if stid in done:
            continue
        try:
            df = get_daily_climate_year(stid, 2017)
            if df is None or df.empty:
                logging.info(f"[{stid}] sin datos 2017.")
                df = pd.DataFrame(columns=["fecha"])
            df["nearest_station"] = stid
            station_frames.append(df)
            _mark_done(stid)
        except Exception as e:
            logging.warning(f"[{stid}] fallo definitivo: {e}")
            _mark_fail(stid)
        _sleep_jitter(1.0, 1.0)  # pausa entre estaciones (1.0–2.0 s)

    clima_station = (pd.concat(station_frames, ignore_index=True)
                     if station_frames else pd.DataFrame(columns=["fecha","nearest_station"]))
    clima_station.to_csv(CLIMATE_2017_STATION_CSV, index=False)

    mapping = valid[["profile_id","nearest_station"]].astype(str)
    clima_profile = mapping.merge(clima_station, on="nearest_station", how="left")
    clima_profile.to_csv(CLIMATE_2017_PROFILE_CSV, index=False)
    logging.info(f"Guardado clima por estación -> {CLIMATE_2017_STATION_CSV}")
    logging.info(f"Guardado clima por perfil   -> {CLIMATE_2017_PROFILE_CSV}")
    return clima_station, clima_profile

def impute_station_climate_2017(clima_station: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa NAs en clima diario por estación:
    1) por (estación, mes) mediana
    2) por (estación) mediana anual
    3) global anual mediana
    """
    if clima_station is None or clima_station.empty:
        return pd.DataFrame(columns=["nearest_station","fecha"])

    df = clima_station.copy()
    if "month" not in df.columns and "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", format="mixed")
        df["month"] = df["fecha"].dt.month

    num_cols = [c for c in df.columns if c not in ("indicativo","nombre","provincia","fecha","month","nearest_station")]
    # 1) estación-mes
    grp1 = df.groupby(["nearest_station","month"])[num_cols].transform("median")
    for c in num_cols:
        df[c] = df[c].fillna(grp1[c])
    # 2) estación anual
    grp2 = df.groupby(["nearest_station"])[num_cols].transform("median")
    for c in num_cols:
        df[c] = df[c].fillna(grp2[c])
    # 3) global
    med_global = df[num_cols].median(numeric_only=True)
    for c in num_cols:
        df[c] = df[c].fillna(med_global[c])

    return df

def build_climate_aggregates(climate_df: pd.DataFrame) -> pd.DataFrame:
    if climate_df is None or climate_df.empty:
        return pd.DataFrame(columns=["profile_id","tmed_mean_2017","tmax_mean_2017","tmin_mean_2017","prec_sum_2017","n_dias_lluvia"])
    df = climate_df.copy()
    aggs = {}
    if "tmed" in df.columns: aggs["tmed_mean_2017"] = ("tmed","mean")
    if "tmax" in df.columns: aggs["tmax_mean_2017"] = ("tmax","mean")
    if "tmin" in df.columns: aggs["tmin_mean_2017"] = ("tmin","mean")
    if "prec" in df.columns:
        aggs["prec_sum_2017"] = ("prec","sum")
        aggs["n_dias_lluvia"] = ("prec", lambda s: (s > 0).sum())
    out = df.groupby("profile_id").agg(**aggs).reset_index()
    return out

# =========================
# Summary dataset final
# =========================
def _infer_var_type(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s): return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s): return "datetime"
    # intenta numérico (coma/pct tolerante)
    s_try = (s.astype(str).str.replace(",", ".", regex=False).str.replace("%","", regex=False))
    num = pd.to_numeric(s_try, errors="coerce")
    if num.notna().mean() > 0.8: return "numeric"
    dt = pd.to_datetime(s, errors="coerce", format="mixed")
    if dt.notna().mean() > 0.8: return "datetime"
    return "categorical"

def make_summary(df: pd.DataFrame, name: str, outdir: Path = OUTPUT_DIR) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for col in df.columns:
        s = df[col]
        t = _infer_var_type(s)
        nn = s.notna().sum()
        miss = s.isna().sum()
        miss_pct = (miss/len(df))*100 if len(df) else 0
        nunique = s.nunique(dropna=True)
        row = dict(variable=str(col), tipo=t, non_null=nn, missing=miss, missing_pct=round(miss_pct,2), nunique=nunique)
        if t == "numeric":
            s_num = pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False).str.replace("%","", regex=False), errors="coerce")
            desc = s_num.describe(percentiles=[.25,.5,.75])
            row.update(dict(mean=desc.get("mean"), std=desc.get("std"),
                            min=desc.get("min"), p25=desc.get("25%"),
                            median=desc.get("50%"), p75=desc.get("75%"),
                            max=desc.get("max")))
        if t == "datetime":
            s_dt = pd.to_datetime(s, errors="coerce", format="mixed")
            if s_dt.notna().any():
                row.update(dict(dt_min=str(s_dt.min()), dt_max=str(s_dt.max())))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / f"summary_{name}.csv", index=False)

    # gráficos sencillos
    try:
        plt.figure(figsize=(max(6, len(df.columns)*0.35), 4))
        miss_counts = df.isna().sum().sort_values(ascending=False)
        miss_counts.plot(kind="bar")
        plt.title(f"Faltantes por variable: {name}")
        plt.ylabel("n faltantes"); plt.tight_layout()
        plt.savefig(outdir / f"missing_{name}.png"); plt.close()
    except Exception as e:
        logging.warning(f"No se pudo graficar faltantes ({name}): {e}")

    # distribuciones rápidas (hasta 6 numéricas)
    num_cols = list(summary[summary["tipo"]=="numeric"]["variable"].head(6))
    for c in num_cols:
        try:
            s_num = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False).str.replace("%","", regex=False), errors="coerce")
            plt.figure(figsize=(6,4))
            sns.histplot(s_num.dropna(), bins=30, kde=True)
            plt.title(f"{name} · {c}"); plt.tight_layout()
            plt.savefig(outdir / f"dist_{name}_{c}.png"); plt.close()
        except Exception as e:
            logging.warning(f"No se pudo graficar {c} ({name}): {e}")

    return summary


def _series_mode(s: pd.Series):
    s2 = s.dropna()
    if s2.empty:
        return np.nan
    m = s2.mode(dropna=True)
    return m.iloc[0] if not m.empty else s2.iloc[0]

def summarize_horizons_to_profile(horizontes: pd.DataFrame) -> pd.DataFrame:
    """
    Resume HORIZONS a nivel profile_id con detección robusta de columnas numéricas.
    - Normaliza nombres de columnas (minúsculas, sin espacios, % ni corchetes).
    - Convierte a numérico siempre que sea posible.
    - Agrega por media y mediana.
    """
    if horizontes is None or horizontes.empty:
        return pd.DataFrame(columns=["profile_id"])

    df = horizontes.copy()
    if "profile_id" not in df.columns:
        logging.warning("Horizons sin 'profile_id': no se puede agregar.")
        return pd.DataFrame(columns=["profile_id"])
    df["profile_id"] = df["profile_id"].astype(str)

    # Normalizar nombres de columnas
    rename_map = {}
    for c in df.columns:
        c_norm = (c.lower()
                    .replace("%", "")
                    .replace("[", "")
                    .replace("]", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("/", "")
                    .replace("*", "")
                    .replace(" ", "_")
                    .strip("_"))
        rename_map[c] = c_norm
    df = df.rename(columns=rename_map)

    # Columnas esperadas (normalizadas)
    expected_numeric = [
        "depth_top_m", "depth_bot_m", "depth_sed_m", "thick_m",
        "density_gcm3", "litho_coarse_material",
        "sand", "silt", "clay_min",
        "om", "toc", "ph", "carb", "c_n",
        "tn_mgkg", "p_mgkg", "k_mgkg", "ca_mgkg",
        "mg_mgkg", "na_mgkg", "cec_cmolkg", "ec_msm", "gp"
    ]

    cleaned_numeric = []
    for col in expected_numeric:
        if col not in df.columns:
            continue
        s = (df[col].astype(str)
                     .str.replace(",", ".", regex=False)
                     .str.replace("%", "", regex=False)
                     .str.strip())
        df[col] = pd.to_numeric(s, errors="coerce")
        cleaned_numeric.append(col)

    if not cleaned_numeric:
        logging.warning("No se detectaron columnas numéricas en horizons tras normalización.")
        return pd.DataFrame(columns=["profile_id"])

    aggs = {col: ["mean", "median"] for col in cleaned_numeric}
    out = df.groupby("profile_id").agg(aggs)
    out.columns = [f"{c}_{stat}" for c, stat in out.columns]
    return out.reset_index()

# =========================
# Indicadores Sprint 2 (EDA avanzado)
# =========================
from typing import Dict, List, Tuple

def _ensure_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False).str.replace("%", "", regex=False),
        errors="coerce"
    )

def compute_missing_rates(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False).to_frame("missing_rate")
    miss["missing_pct"] = (miss["missing_rate"]*100).round(2)
    miss.reset_index(names="variable", inplace=True)
    return miss

def compute_outlier_rate_iqr(df: pd.DataFrame, min_numeric_ratio: float = 0.8) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s_num = _ensure_numeric_series(df[c])
        if s_num.notna().mean() < min_numeric_ratio:
            continue
        q1, q3 = s_num.quantile([.25, .75])
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            out_rate = 0.0
        else:
            lb, ub = q1 - 1.5*iqr, q3 + 1.5*iqr
            out_rate = ((s_num < lb) | (s_num > ub)).mean()
        rows.append({"variable": c, "outlier_rate": out_rate, "outlier_pct": round(out_rate*100, 2)})
    return pd.DataFrame(rows).sort_values("outlier_rate", ascending=False)

def type_consistency_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        t = _infer_var_type(s)  # ya la tienes definida
        num_try = _ensure_numeric_series(s)
        numeric_ratio = num_try.notna().mean()
        dt_try = pd.to_datetime(s, errors="coerce", format="mixed")
        dt_ratio = dt_try.notna().mean()
        rows.append({
            "variable": c,
            "inferred_type": t,
            "numeric_coercible_pct": round(100*numeric_ratio, 2),
            "datetime_coercible_pct": round(100*dt_ratio, 2),
            "nunique": s.nunique(dropna=True),
            "missing_pct": round(100*s.isna().mean(), 2),
        })
    return pd.DataFrame(rows).sort_values(["inferred_type","missing_pct","variable"])

# ---------- PSI (Population Stability Index) ----------
def _psi_bins_from_quantiles(base: pd.Series, bins: int = 10) -> np.ndarray:
    s = _ensure_numeric_series(base).dropna()
    if s.empty:
        return np.array([])
    qs = np.linspace(0, 1, bins+1)
    edges = np.unique(s.quantile(qs).values)
    # asegurar límites
    if len(edges) < 3:
        edges = np.unique(np.concatenate([[s.min()-1e-9], [s.median()], [s.max()+1e-9]]))
    edges[0] = -np.inf; edges[-1] = np.inf
    return edges

def _psi_from_counts(base_counts: np.ndarray, comp_counts: np.ndarray) -> float:
    # suavizado para evitar div/0
    base_ratio = base_counts / (base_counts.sum() + 1e-12)
    comp_ratio = comp_counts / (comp_counts.sum() + 1e-12)
    base_ratio = np.maximum(base_ratio, 1e-6)
    comp_ratio = np.maximum(comp_ratio, 1e-6)
    return float(np.sum((comp_ratio - base_ratio) * np.log(comp_ratio / base_ratio)))

def psi_numeric(base: pd.Series, comp: pd.Series, bins: int = 10) -> float:
    edges = _psi_bins_from_quantiles(base, bins=bins)
    if edges.size < 3:
        return np.nan
    b = pd.cut(_ensure_numeric_series(base), bins=edges).value_counts(sort=False).values
    c = pd.cut(_ensure_numeric_series(comp), bins=edges).value_counts(sort=False).values
    return _psi_from_counts(b, c)

def psi_categorical(base: pd.Series, comp: pd.Series) -> float:
    bvc = base.astype(str).value_counts()
    cvc = comp.astype(str).value_counts()
    cats = sorted(set(bvc.index).union(cvc.index))
    b = np.array([bvc.get(k, 0) for k in cats], dtype=float)
    c = np.array([cvc.get(k, 0) for k in cats], dtype=float)
    return _psi_from_counts(b, c)

# ---------- Visualizaciones ----------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def plot_missing_heatmap(df: pd.DataFrame, outdir: Path, name="missing_heatmap.png", max_cols: int = 60):
    try:
        plt.figure(figsize=(min(24, max(8, df.shape[1]*0.25)), 10))
        sns.heatmap(df.iloc[:, :max_cols].isna(), cbar=False)
        plt.title("Mapa de valores faltantes (primeras columnas)")
        plt.tight_layout()
        plt.savefig(outdir / name); plt.close()
    except Exception as e:
        logging.warning(f"No se pudo crear heatmap de faltantes: {e}")

def plot_boxplots(df: pd.DataFrame, outdir: Path, numeric_cols: List[str], max_plots: int = 12):
    sel = numeric_cols[:max_plots]
    for c in sel:
        try:
            s = _ensure_numeric_series(df[c])
            plt.figure(figsize=(6,4))
            sns.boxplot(x=s.dropna())
            plt.title(f"Boxplot · {c}")
            plt.tight_layout()
            plt.savefig(outdir / f"box_{c}.png"); plt.close()
        except Exception as e:
            logging.warning(f"No se pudo boxplot {c}: {e}")

def plot_histograms(df: pd.DataFrame, outdir: Path, numeric_cols: List[str], max_plots: int = 12):
    sel = numeric_cols[:max_plots]
    for c in sel:
        try:
            s = _ensure_numeric_series(df[c])
            plt.figure(figsize=(6,4))
            sns.histplot(s.dropna(), bins=30, kde=True)
            plt.title(f"Hist · {c}")
            plt.tight_layout()
            plt.savefig(outdir / f"hist_{c}.png"); plt.close()
        except Exception as e:
            logging.warning(f"No se pudo hist {c}: {e}")

def plot_target_balance(df: pd.DataFrame, target: str, outdir: Path, top_n: int = 20):
    vc = df[target].astype(str).value_counts().head(top_n)
    plt.figure(figsize=(10,5))
    sns.barplot(x=vc.index, y=vc.values)
    plt.title(f"Balance de clases · {target}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / f"balance_{target}.png"); plt.close()

def plot_target_by_group(df: pd.DataFrame, target: str, group_col: str, outdir: Path, top_groups: int = 20):
    grp = (df.groupby(group_col)[target]
           .value_counts(normalize=True)
           .rename("pct")
           .reset_index())
    top = df[group_col].astype(str).value_counts().head(top_groups).index
    grp = grp[grp[group_col].astype(str).isin(top)]
    g = sns.catplot(data=grp, x=group_col, y="pct", hue=target, kind="bar", height=5, aspect=2)
    g.set_xticklabels(rotation=45, ha="right")
    plt.title(f"Distribución de {target} por {group_col} (Top {top_groups})")
    plt.tight_layout()
    plt.savefig(outdir / f"{target}_por_{group_col}.png"); plt.close()

# ---------- Orquestador principal del análisis ----------
def analyze_dataset(final_df: pd.DataFrame, outdir: Path = OUTPUT_DIR / "analysis", target_col: str = "cultivo_grupo"):
    outdir = _ensure_dir(outdir)

    # 0) Tipos e identificación de variables candidatas
    type_rep = type_consistency_report(final_df)
    type_rep.to_csv(outdir / "types_consistency.csv", index=False)

    # 1) Missing
    miss = compute_missing_rates(final_df)
    miss.to_csv(outdir / "missing_rates.csv", index=False)
    plot_missing_heatmap(final_df, outdir)

    # 2) Outliers (IQR) sobre numéricas "fiables"
    numeric_cols = []
    for c in final_df.columns:
        s_num = _ensure_numeric_series(final_df[c])
        if s_num.notna().mean() >= 0.8:
            numeric_cols.append(c)

    outliers = compute_outlier_rate_iqr(final_df)
    outliers.to_csv(outdir / "outlier_rates_iqr.csv", index=False)

    # 3) Balance de clases (+ cobertura por grupo de cultivo)
    if target_col in final_df.columns:
        plot_target_balance(final_df, target_col, outdir)
        coverage = final_df[target_col].value_counts(dropna=False).rename_axis(target_col).reset_index(name="n")
        coverage["pct"] = (100*coverage["n"]/len(final_df)).round(2)
        coverage.to_csv(outdir / "target_balance.csv", index=False)

        # estabilidad de la tasa objetivo por provincia (si existe)
        group_col = "station_provincia" if "station_provincia" in final_df.columns else None
        if group_col:
            plot_target_by_group(final_df, target_col, group_col, outdir)
            stab = (final_df.groupby(group_col)[target_col]
                    .value_counts(normalize=True)
                    .rename("rate")
                    .reset_index())
            stab.to_csv(outdir / f"target_rate_by_{group_col}.csv", index=False)

    # 4) Hist/Box de variables climáticas y numéricas clave
    clima_pref = [c for c in final_df.columns if c.endswith("_2017") or c in ["n_dias_lluvia"]]
    to_plot = [c for c in clima_pref if c in numeric_cols] or numeric_cols[:8]
    plot_histograms(final_df, outdir, to_plot, max_plots=min(12, len(to_plot)))
    plot_boxplots(final_df, outdir, to_plot, max_plots=min(12, len(to_plot)))

    # 5) PSI: comparamos por provincia vs. global (numéricas y target)
    if "station_provincia" in final_df.columns:
        prov = final_df["station_provincia"].astype(str)
        psi_rows = []

        # PSI numérico (contra global) para clima
        for c in clima_pref:
            if c not in final_df.columns: 
                continue
            base = final_df[c]
            for pv, dfp in final_df.groupby(prov):
                comp = dfp[c]
                val = psi_numeric(base, comp, bins=10)
                psi_rows.append({"variable": c, "group": pv, "psi": val})

        # PSI categórico para el target (si lo hay)
        if target_col in final_df.columns:
            base_t = final_df[target_col]
            for pv, dfp in final_df.groupby(prov):
                comp_t = dfp[target_col]
                val_t = psi_categorical(base_t, comp_t)
                psi_rows.append({"variable": f"PSI_{target_col}", "group": pv, "psi": val_t})

        psi_df = pd.DataFrame(psi_rows).sort_values(["variable","psi"], ascending=[True, False])
        psi_df.to_csv(outdir / "psi_by_provincia.csv", index=False)

    # 6) Matriz de correlación (numéricas)
    num_for_corr = []
    for c in final_df.columns:
        s = _ensure_numeric_series(final_df[c])
        if s.notna().mean() >= 0.8:
            final_df[f"__num__{c}"] = s
            num_for_corr.append(f"__num__{c}")
    if num_for_corr:
        try:
            corr = final_df[num_for_corr].corr(numeric_only=True)
            plt.figure(figsize=(min(18, 0.4*len(num_for_corr)+6), min(18, 0.4*len(num_for_corr)+6)))
            sns.heatmap(corr, cmap="vlag", center=0)
            plt.title("Matriz de correlación (numéricas)")
            plt.tight_layout()
            plt.savefig(outdir / "corr_matrix.png"); plt.close()
        except Exception as e:
            logging.warning(f"No se pudo crear matriz de correlación: {e}")
        finally:
            final_df.drop(columns=num_for_corr, inplace=True, errors="ignore")

    # 7) Resumen ejecutivo (CSV)
    #   - tasa global de missing
    miss_global = round(100*final_df.isna().mean().mean(), 2)
    #   - media de outliers (sobre variables medidas)
    out_mean = round(100*outliers["outlier_rate"].mean(), 2) if not outliers.empty else 0.0
    #   - cobertura clima (si existen)
    cov_items = []
    for c in clima_pref:
        if c in final_df.columns:
            cov_items.append((c, round(100*final_df[c].notna().mean(), 2)))
    rep = pd.DataFrame({
        "metric": ["missing_global_pct", "outliers_mean_pct"] + [f"cov_{k}" for k,_ in cov_items],
        "value":  [miss_global, out_mean] + [v for _,v in cov_items]
    })
    rep.to_csv(outdir / "analysis_report.csv", index=False)
    logging.info("Análisis Sprint 2 generado en %s", str(outdir))

from datetime import datetime

def safe_to_csv(df, path, **kwargs):
    """
    Guarda un DataFrame en CSV. 
    Si el archivo está bloqueado (ej. abierto en Excel), crea una copia con timestamp.
    """
    try:
        df.to_csv(path, **kwargs)
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        import logging
        logging.warning(f"Archivo bloqueado: {path.name}. Guardando como {alt.name}")
        df.to_csv(alt, **kwargs)

# =========================
# MAIN
# =========================
def main():
    prof_path = DATOS_DIR / "CARBOSOL_profile.tab"
    horiz_path = DATOS_DIR / "CARBOSOL_horizons.tab"

    # 1) Cargar CARBOSOL (con cabecera robusta)
    perfiles = _read_pangaea_tab(prof_path)
    horizontes = _read_pangaea_tab(horiz_path)
    logging.info(f"{prof_path.name} columnas (primeras 20): {list(perfiles.columns)[:20]}")
    logging.info(f"{horiz_path.name} columnas (primeras 20): {list(horizontes.columns)[:20]}")

    # 2) Resolver IDs y coords
    perfiles, horizontes = _resolve_ids(perfiles, horizontes)
    logging.info(f"Perfiles -> {perfiles.shape}, cols: {list(perfiles.columns)[:20]}...")
    logging.info(f"Horizontes -> {horizontes.shape}, cols: {list(horizontes.columns)[:20]}...")

    # 3) Variable objetivo: cultivo = Description(...)
    perfiles = _ensure_cultivo_from_description(perfiles)

    # 3.b) Conteo de perfiles sin cultivo
    sin_cultivo = perfiles["cultivo"].astype(str).str.strip().replace({"": np.nan}).isna().sum()
    with open(OUTPUT_DIR / "cultivo_missing.txt", "w", encoding="utf-8") as f:
        f.write(f"Perfiles SIN cultivo (vacío/NA): {sin_cultivo} de {len(perfiles)}\n")
    logging.info(f"Perfiles SIN cultivo: {sin_cultivo}/{len(perfiles)} (ver cultivo_missing.txt)")

    # 3.c) NUEVO: Agrupación de cultivo con reglas sólidas
    perfiles["cultivo_grupo"] = perfiles["cultivo"].astype(str).apply(assign_cultivo_group)

    # Guarda frecuencias de crudos y de grupos
    (perfiles["cultivo"].astype(str).fillna("NA").value_counts(dropna=False)
     .rename_axis("cultivo").reset_index(name="n")
     .to_csv(OUTPUT_DIR / "cultivo_frecuencias.csv", index=False))

    (perfiles["cultivo_grupo"].astype(str).fillna("NA").value_counts(dropna=False)
     .rename_axis("cultivo_grupo").reset_index(name="n")
     .to_csv(OUTPUT_DIR / "cultivo_grupo_frecuencias.csv", index=False))

    otros_top = (perfiles.loc[perfiles["cultivo_grupo"]=="Otros","cultivo"]
             .astype(str).str.strip().value_counts().head(150).reset_index())
    otros_top.columns = ["cultivo_descripcion","n"]
    safe_to_csv(otros_top, OUTPUT_DIR / "cultivo_otros_top150.csv", index=False, encoding="utf-8-sig")

    # 4) Estaciones AEMET y asignación SIEMPRE
    stations = get_aemet_stations()
    assign = assign_nearest_station(perfiles, stations)
    perfiles = perfiles.merge(assign, on="profile_id", how="left")
    logging.info(f"Estaciones asignadas a perfiles: {perfiles['nearest_station'].notna().sum()}/{len(perfiles)}")

    # 5) Clima 2017: lee CSV si existe; si no, descarga y guarda
    if CLIMATE_2017_STATION_CSV.exists() and CLIMATE_2017_PROFILE_CSV.exists():
        logging.info("Leyendo clima 2017 desde CSV guardado")
        clima_station = pd.read_csv(CLIMATE_2017_STATION_CSV)
        climate_df = pd.read_csv(CLIMATE_2017_PROFILE_CSV)
    else:
        logging.info("Descargando clima 2017 y guardando CSVs…")
        clima_station, climate_df = build_and_save_climate_2017(perfiles)

    # 6) IMPUTACIÓN de clima a nivel estación y proyección a perfil
    if not clima_station.empty:
        clima_station_imp = impute_station_climate_2017(clima_station)
        clima_station_imp.to_csv(CLIMATE_2017_STATION_IMPUTED_CSV, index=False)

        mapping = perfiles[["profile_id","nearest_station"]].astype(str)
        climate_df_imp = mapping.merge(clima_station_imp, on="nearest_station", how="left")
        climate_df_imp.to_csv(CLIMATE_2017_PROFILE_IMPUTED_CSV, index=False)
    else:
        climate_df_imp = pd.DataFrame(columns=["profile_id"])

    # Tipos en climate_df_imp
    if not climate_df_imp.empty:
        if "fecha" in climate_df_imp.columns:
            climate_df_imp["fecha"] = pd.to_datetime(climate_df_imp["fecha"], errors="coerce", format="mixed")
        for c in ["tmed","tmax","tmin","prec","tpr"]:
            if c in climate_df_imp.columns:
                climate_df_imp[c] = pd.to_numeric(climate_df_imp[c], errors="coerce")
        if "profile_id" in climate_df_imp.columns:
            climate_df_imp["profile_id"] = climate_df_imp["profile_id"].astype(str)

    # 7) Agregados climáticos por perfil (con clima imputado)
    climate_agg = build_climate_aggregates(climate_df_imp)

    # === NUEVO: resumen de horizons a nivel perfil ===
    hz_summary = summarize_horizons_to_profile(horizontes)

    # 8) Dataset final (TODAS las variables de profile + horizons agregados + clima agregado)
    #    Nota: 'perfiles' ya incluye 'cultivo' y 'cultivo_grupo' (objetivo)
    perfiles_all = perfiles.copy()
    if "profile_id" in perfiles_all.columns:
        perfiles_all["profile_id"] = perfiles_all["profile_id"].astype(str)

    if not hz_summary.empty:
        hz_summary["profile_id"] = hz_summary["profile_id"].astype(str)

    if not climate_agg.empty and "profile_id" in climate_agg.columns:
        climate_agg["profile_id"] = climate_agg["profile_id"].astype(str)

    final_df = perfiles_all.merge(hz_summary, on="profile_id", how="left") \
                           .merge(climate_agg, on="profile_id", how="left")

    final_df.to_csv(OUTPUT_DIR / "dataset_final_2017_full.csv", index=False)

    # 8.b) Derivar dataset para el MODELO (filtrado sin 'Otros')
    final_df_model = final_df[final_df["cultivo_grupo"] != "Otros"].copy()
    MODEL_OUTDIR = OUTPUT_DIR / "model"
    MODEL_OUTDIR.mkdir(parents=True, exist_ok=True)
    safe_to_csv(final_df_model, MODEL_OUTDIR / "dataset_final_2017_model.csv", index=False, encoding="utf-8-sig")


    # 9) Summary del dataset final (tablas + gráficos)
    make_summary(final_df, "dataset_final_2017_full")

    # === Análisis Sprint 2 ===
    try:
        analyze_dataset(
            final_df,
            outdir=OUTPUT_DIR / "analysis",
            target_col="cultivo_grupo"
        )
    except Exception as e:
        logging.error(f"Error en análisis Sprint 2: {e}")

    # 10) Mapa (opcional)
    try:
        m = folium.Map(location=[40.4, -3.7], zoom_start=5)
        for _, r in perfiles.dropna(subset=["lat","lon"]).iterrows():
            folium.Marker([r["lat"], r["lon"]],
                          popup=f"Perfil: {r['profile_id']}<br>Cultivo: {r.get('cultivo','-')}<br>Grupo: {r.get('cultivo_grupo','-')}<br>Est: {r.get('nearest_station','-')}"
                          ).add_to(m)
        m.save(OUTPUT_DIR / "perfiles_map.html")
    except Exception as e:
        logging.warning(f"No se pudo crear el mapa: {e}")

    logging.info("Pipeline completado. Revisa outputs/eda/")

if __name__ == "__main__":
    main()

