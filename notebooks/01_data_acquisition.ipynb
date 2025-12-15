"""
# 01 — Ingestion & Feature Engineering (CARBOSOL + AEMET)

**Objetivo**
- Construir un dataset a nivel **perfil** uniendo:
  - CARBOSOL profiles + horizons
  - Variable objetivo a partir de **Description → cultivo**
  - Agrupación avanzada de cultivo (**cultivo_grupo**)
  - AEMET clima diario 2017 (estación más cercana) + imputación + agregación anual

**Artefactos**
- `outputs/eda/dataset_final_2017_full.csv`
- `outputs/eda/model/dataset_final_2017_model.csv` (sin clase "Otros")
- Checks intermedios en `outputs/eda/` y cache en `cache/`
"""

import os, re, json, math, time, random, logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from http.client import RemoteDisconnected
from json import JSONDecodeError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
pd.set_option("display.max_columns", 120)

# =========================
# Configuración y rutas
# =========================
PROJECT_DIR = Path("..").resolve() if (Path.cwd().name == "notebooks") else Path(".").resolve()

DATA_DIR   = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "eda"
CACHE_DIR  = PROJECT_DIR / "cache"

for p in (OUTPUT_DIR, CACHE_DIR, OUTPUT_DIR / "model"):
    p.mkdir(parents=True, exist_ok=True)

# Archivos CARBOSOL (esperados en data/)
CARBOSOL_PROFILE  = DATA_DIR / "CARBOSOL_profile.tab"
CARBOSOL_HORIZONS = DATA_DIR / "CARBOSOL_horizons.tab"

# CSVs clima
CLIMATE_2017_STATION_CSV          = OUTPUT_DIR / "clima_2017_por_estacion.csv"
CLIMATE_2017_STATION_IMPUTED_CSV  = OUTPUT_DIR / "clima_2017_por_estacion_imputado.csv"
CLIMATE_2017_PROFILE_CSV          = OUTPUT_DIR / "clima_2017_por_perfil.csv"
CLIMATE_2017_PROFILE_IMPUTED_CSV  = OUTPUT_DIR / "clima_2017_por_perfil_imputado.csv"

# API KEY AEMET (NO hardcode)
# opción A: export AEMET_API_KEY="..."
# opción B: usar python-dotenv local (si quieres) -> .env (ignorado)
API_KEY = os.getenv("AEMET_API_KEY", "").strip()
if not API_KEY:
    logging.warning("AEMET_API_KEY no encontrada en variables de entorno. La descarga AEMET fallará.")

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
        if raw.startswith("/"): 
            continue
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
    df = pd.read_csv(path, sep="\t", header=hdr, engine="python", on_bad_lines="skip", dtype=str)

    # elimina columnas 'Unnamed'
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed", na=False)].copy()

    # quita primera fila si parecen unidades
    if len(df) > 0:
        sample = " ".join(map(str, df.iloc[0, :15].tolist()))
        if any(sym in sample for sym in ("[", "]", "°", "‰")):
            df = df.iloc[1:].reset_index(drop=True)

    return df

def _resolve_ids(perfiles: pd.DataFrame, horizontes: pd.DataFrame):
    pf = perfiles.copy()
    hz = horizontes.copy()

    # profile_id en perfiles = Sample ID SIN sufijo ".1"
    pf_cols = [c for c in pf.columns if c.lower().startswith("sample id (unique identification number") and not c.endswith(".1")]
    if pf_cols:
        pf = pf.rename(columns={pf_cols[0]: "profile_id"})
    elif "profile_id" not in pf.columns:
        pf["profile_id"] = np.arange(len(pf)).astype(str)
        logging.warning("No se encontró columna Profile ID en 'profile'; se creó sintética.")

    # profile_id en horizontes = Sample ID con sufijo ".1"
    hz_cols = [c for c in hz.columns if c.lower().startswith("sample id (unique identification number") and c.endswith(".1")]
    if hz_cols:
        hz = hz.rename(columns={hz_cols[0]: "profile_id"})
    elif "profile_id" not in hz.columns:
        logging.warning("No se encontró Profile ID en 'horizons'; joins pueden ser incompletos.")

    # lat/lon
    def _push_latlon(df):
        for name in list(df.columns):
            low = name.lower()
            if low == "latitude" or name == "Latitude":
                df["lat"] = df[name].map(_coerce_decimal)
            if low == "longitude" or name == "Longitude":
                df["lon"] = df[name].map(_coerce_decimal)
        return df

    pf = _push_latlon(pf)
    hz = _push_latlon(hz)

    pf["profile_id"] = pf["profile_id"].astype(str)
    if "profile_id" in hz.columns:
        hz["profile_id"] = hz["profile_id"].astype(str)

    return pf, hz

def _ensure_cultivo_from_description(perfiles: pd.DataFrame) -> pd.DataFrame:
    pf = perfiles.copy()

    cand = [c for c in pf.columns if "description" in c.lower()]
    pref = [c for c in cand if "vegetation" in c.lower() and "provided" in c.lower()]
    col = pref[0] if pref else (cand[0] if cand else None)
    if not col:
        raise KeyError("No se encontró la columna de Description en 'profile'.")

    pf["cultivo"] = pf[col].astype(str).str.strip()

    cats = (pf["cultivo"].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique())
    pd.Series(np.sort(cats), name="categoria_cultivo").to_csv(OUTPUT_DIR / "cultivo_categorias_unicas.csv", index=False)
    logging.info(f"Columna 'cultivo' creada desde: {col}. Categorías únicas: {len(cats)}")

    return pf

import unicodedata

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

_DNUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

def _parse_aemet_dms(s: str) -> float:
    if pd.isna(s): 
        return np.nan
    t = str(s).strip().upper()
    if not t:
        return np.nan

    hem = None
    if t[-1] in "NSEW":
        hem = t[-1]
        t = t[:-1].strip()

    t_norm = (t.replace("º", " ").replace("°", " ")
                .replace("’", " ").replace("'", " ")
                .replace("″", " ").replace('"', " ")
                .replace("  ", " ").strip())

    only_digits = re.sub(r"\D", "", t)
    if only_digits and t_norm.replace(" ", "") == only_digits:
        L = len(only_digits)
        if L >= 4:
            if L >= 6:
                sec = float(only_digits[-2:])
                minu = float(only_digits[-4:-2])
                deg  = float(only_digits[:-4])
            else:
                sec = 0.0
                minu = float(only_digits[-2:])
                deg  = float(only_digits[:-2])
            val = deg + minu/60.0 + sec/3600.0
            if hem in ("S","W"): val = -abs(val)
            elif hem in ("N","E"): val = abs(val)
            return val if (-180 <= val <= 180) else np.nan

    try:
        val = float(t_norm.replace(",", "."))
        if hem in ("S","W"): val = -abs(val)
        elif hem in ("N","E"): val = abs(val)
        return val if (-180 <= val <= 180) else np.nan
    except:
        pass

    nums = _DNUM.findall(t_norm)
    if len(nums) >= 1:
        deg = float(nums[0].replace(",", "."))
        minu = float(nums[1].replace(",", ".")) if len(nums) >= 2 else 0.0
        sec  = float(nums[2].replace(",", ".")) if len(nums) >= 3 else 0.0
        val = deg + minu/60.0 + sec/3600.0
        if hem in ("S","W"): val = -abs(val)
        elif hem in ("N","E"): val = abs(val)
        return val if (-180 <= val <= 180) else np.nan

    return np.nan

# -------- session + retries ----------
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
    if _SESSION is None or _SESSION_USES >= 25:
        _SESSION = _new_session()
        _SESSION_USES = 0
    _SESSION_USES += 1
    return _SESSION

def _sleep_jitter(base=0.8, spread=0.8):
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

def _ensure_datetime(s):
    if not pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce", format="mixed")
    return s

def get_aemet_stations() -> pd.DataFrame:
    if not API_KEY:
        raise RuntimeError("Falta AEMET_API_KEY en entorno. No se puede descargar inventario estaciones.")

    cache_file = CACHE_DIR / "stations.json"

    def _load():
        url = "https://opendata.aemet.es/opendata/api/valores/climatologicos/inventarioestaciones/todasestaciones"
        headers = {"api_key": API_KEY}
        meta = _get_json(url, headers=headers, timeout=45, tries=5)
        payload = _get_json(meta["datos"], headers=headers, timeout=60, tries=6)
        df = pd.DataFrame(payload)
        df.to_json(cache_file, orient="records")
        return df

    df = pd.read_json(cache_file, orient="records") if cache_file.exists() else _load()

    df["lat"] = df["latitud"].apply(_parse_aemet_dms)
    df["lon"] = df["longitud"].apply(_parse_aemet_dms)
    df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)].copy()

    for c in ("fechaAlta", "fechaBaja"):
        df[c] = pd.to_datetime(df.get(c, pd.NaT), errors="coerce", format="mixed")

    # estaciones activas en 2017
    y0 = pd.Timestamp("2017-01-01")
    y1 = pd.Timestamp("2017-12-31 23:59:59")
    alta_ok = (df["fechaAlta"].isna()) | (df["fechaAlta"] <= y1)
    baja_ok = (df["fechaBaja"].isna()) | (df["fechaBaja"] >= y0)
    df = df[alta_ok & baja_ok].copy()

    logging.info(f"Inventario estaciones activas en 2017: {len(df)}")
    return df.reset_index(drop=True)

def _active_mask_for_year(stations: pd.DataFrame, year: int) -> pd.Series:
    st = stations.copy()
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
    pf = perfiles.copy()

    if "lat" not in pf.columns or "lon" not in pf.columns:
        raise ValueError("No se encontraron columnas lat/lon en perfiles.")

    pf["lat"] = pd.to_numeric(pf["lat"], errors="coerce")
    pf["lon"] = pd.to_numeric(pf["lon"], errors="coerce")

    pfv = pf.dropna(subset=["lat","lon"]).copy()
    if pfv.empty:
        raise ValueError("Ningún perfil tiene lat/lon válidos.")

    st = stations.dropna(subset=["lat","lon"]).copy()
    if st.empty:
        raise ValueError("No hay estaciones con lat/lon válidos.")

    active_mask = _active_mask_for_year(st, year).to_numpy()

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

    order = np.argsort(dist_km, axis=1)

    chosen_idx = np.empty(order.shape[0], dtype=int)
    chosen_dist = np.empty(order.shape[0], dtype=float)
    chosen_rank = np.empty(order.shape[0], dtype=int)
    chosen_active = np.empty(order.shape[0], dtype=bool)

    for i in range(order.shape[0]):
        ord_i = order[i]
        active_ord = active_mask[ord_i]
        if active_ord.any():
            k = int(np.argmax(active_ord))
            j = ord_i[k]
            chosen_idx[i] = j
            chosen_dist[i] = float(dist_km[i, j])
            chosen_rank[i] = k + 1
            chosen_active[i] = True
        else:
            j = ord_i[0]
            chosen_idx[i] = j
            chosen_dist[i] = float(dist_km[i, j])
            chosen_rank[i] = 1
            chosen_active[i] = False

    st_ix = st.reset_index(drop=True).iloc[chosen_idx]

    out = pfv[["profile_id"]].copy()
    out["nearest_station"]     = st_ix["indicativo"].astype(str).values
    out["station_name"]        = st_ix.get("nombre", pd.Series([""]*len(out))).astype(str).values
    out["station_provincia"]   = st_ix.get("provincia", pd.Series([""]*len(out))).astype(str).values
    out["distance_km"]         = chosen_dist
    out["station_active_2017"] = chosen_active
    out["station_rank_used"]   = chosen_rank

    out.to_csv(OUTPUT_DIR / "perfiles_estaciones_check.csv", index=False)
    logging.info("Guardado check asignación estaciones: outputs/eda/perfiles_estaciones_check.csv")
    return out

def _clean_daily_payload(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: 
        return df
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", format="mixed")
        df["month"] = df["fecha"].dt.month

    numeric_cols = [c for c in df.columns if c not in ("indicativo","nombre","provincia","fecha","month")]
    for col in numeric_cols:
        s = df[col].astype(str).str.strip()
        if col.lower().startswith("prec"):
            s = s.replace({"Ip":"0","ip":"0"})
        s = s.str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(s, errors="coerce")
    return df

def _range_url(station_id: str, start: str, end: str) -> str:
    return ("https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/"
            f"datos/fechaini/{start}T00:00:00UTC/fechafin/{end}T23:59:59UTC/estacion/{station_id}")

def _fetch_range(stid: str, start: str, end: str, headers: dict, meta_tries=3) -> pd.DataFrame:
    meta_url = _range_url(stid, start, end)
    meta = _get_json(meta_url, headers=headers, timeout=45, tries=5)
    if not meta or "datos" not in meta or not meta.get("datos"):
        return pd.DataFrame()

    short = meta["datos"]
    for k in range(meta_tries):
        try:
            payload = _get_json(short, headers=headers, timeout=60, tries=6)
            if payload:
                return _clean_daily_payload(pd.DataFrame(payload))
        except Exception as e:
            logging.warning(f"[{stid} {start}..{end}] short URL falló (k={k+1}): {e}")
            meta = _get_json(meta_url, headers=headers, timeout=45, tries=5)
            short = meta.get("datos")
        time.sleep(0.4 + random.random()*0.5)
    return pd.DataFrame()

def get_daily_climate_year(stid: str, year=2017) -> pd.DataFrame:
    if not API_KEY:
        raise RuntimeError("Falta AEMET_API_KEY. No se puede descargar clima.")

    cache_file = CACHE_DIR / f"climate_{stid}_{year}.json"
    headers = {"api_key": API_KEY}

    if cache_file.exists():
        try:
            return _clean_daily_payload(pd.read_json(cache_file, orient="records"))
        except Exception:
            pass

    ranges = [
        [(f"{year}-01-01", f"{year}-12-31")],
        [(f"{year}-01-01", f"{year}-06-30"), (f"{year}-07-01", f"{year}-12-31")],
        [(f"{year}-01-01", f"{year}-03-31"), (f"{year}-04-01", f"{year}-06-30"),
         (f"{year}-07-01", f"{year}-09-30"), (f"{year}-10-01", f"{year}-12-31")],
        [(f"{year}-{m:02d}-01", f"{year}-{m:02d}-{_last_day_of_month(year,m):02d}") for m in range(1,13)],
    ]

    frames = []
    got_any = False
    for rr in ranges:
        frames = []
        got_any = False
        for (start, end) in rr:
            df = _fetch_range(stid, start, end, headers=headers, meta_tries=3)
            if not df.empty:
                frames.append(df); got_any = True
            _sleep_jitter(0.3, 0.5)
        if got_any:
            break

    if not got_any:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out.to_json(cache_file, orient="records")
    return out

DONE_FILE = CACHE_DIR / "climate_2017_done.txt"
FAIL_FILE = CACHE_DIR / "climate_2017_failed.txt"

def _load_done():
    if DONE_FILE.exists():
        return set(x.strip() for x in DONE_FILE.read_text(encoding="utf-8").splitlines() if x.strip())
    return set()

def _mark_done(stid):
    with open(DONE_FILE, "a", encoding="utf-8") as f:
        f.write(stid + "\n")

def _mark_fail(stid):
    with open(FAIL_FILE, "a", encoding="utf-8") as f:
        f.write(stid + "\n")

def build_and_save_climate_2017(perfiles_with_station: pd.DataFrame):
    valid = perfiles_with_station.dropna(subset=["nearest_station"]).copy()
    unique_stations = list(pd.unique(valid["nearest_station"].astype(str)))
    done = _load_done()

    station_frames = []
    for stid in unique_stations:
        if stid in done:
            continue
        try:
            df = get_daily_climate_year(stid, 2017)
            if df is None or df.empty:
                df = pd.DataFrame(columns=["fecha"])
            df["nearest_station"] = stid
            station_frames.append(df)
            _mark_done(stid)
        except Exception as e:
            logging.warning(f"[{stid}] fallo definitivo: {e}")
            _mark_fail(stid)
        _sleep_jitter(1.0, 1.0)

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
    if clima_station is None or clima_station.empty:
        return pd.DataFrame(columns=["nearest_station","fecha"])

    df = clima_station.copy()
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", format="mixed")
        df["month"] = df["fecha"].dt.month

    num_cols = [c for c in df.columns if c not in ("indicativo","nombre","provincia","fecha","month","nearest_station")]

    grp1 = df.groupby(["nearest_station","month"])[num_cols].transform("median")
    for c in num_cols:
        df[c] = df[c].fillna(grp1[c])

    grp2 = df.groupby(["nearest_station"])[num_cols].transform("median")
    for c in num_cols:
        df[c] = df[c].fillna(grp2[c])

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

def summarize_horizons_to_profile(horizontes: pd.DataFrame) -> pd.DataFrame:
    if horizontes is None or horizontes.empty:
        return pd.DataFrame(columns=["profile_id"])

    df = horizontes.copy()
    if "profile_id" not in df.columns:
        logging.warning("Horizons sin profile_id.")
        return pd.DataFrame(columns=["profile_id"])

    df["profile_id"] = df["profile_id"].astype(str)

    # normaliza nombres
    rename_map = {}
    for c in df.columns:
        c_norm = (c.lower()
                    .replace("%","")
                    .replace("[","").replace("]","")
                    .replace("(","").replace(")","")
                    .replace("/","").replace("*","")
                    .replace(" ","_")
                    .strip("_"))
        rename_map[c] = c_norm
    df = df.rename(columns=rename_map)

    expected_numeric = [
        "depth_top_m","depth_bot_m","depth_sed_m","thick_m",
        "density_gcm3","litho_coarse_material",
        "sand","silt","clay_min",
        "om","toc","ph","carb","c_n",
        "tn_mgkg","p_mgkg","k_mgkg","ca_mgkg",
        "mg_mgkg","na_mgkg","cec_cmolkg","ec_msm","gp"
    ]

    cleaned_numeric = []
    for col in expected_numeric:
        if col not in df.columns:
            continue
        s = (df[col].astype(str).str.replace(",", ".", regex=False).str.replace("%","", regex=False).str.strip())
        df[col] = pd.to_numeric(s, errors="coerce")
        cleaned_numeric.append(col)

    if not cleaned_numeric:
        logging.warning("No se detectaron columnas numéricas en horizons tras normalización.")
        return pd.DataFrame(columns=["profile_id"])

    aggs = {col: ["mean","median"] for col in cleaned_numeric}
    out = df.groupby("profile_id").agg(aggs)
    out.columns = [f"{c}_{stat}" for c, stat in out.columns]
    return out.reset_index()

def safe_to_csv(df, path, **kwargs):
    try:
        df.to_csv(path, **kwargs)
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        logging.warning(f"Archivo bloqueado: {path.name}. Guardando como {alt.name}")
        df.to_csv(alt, **kwargs)

# =========================
# MAIN (ejecución)
# =========================

# 1) Cargar CARBOSOL
assert CARBOSOL_PROFILE.exists(), f"Falta {CARBOSOL_PROFILE}"
assert CARBOSOL_HORIZONS.exists(), f"Falta {CARBOSOL_HORIZONS}"

perfiles = _read_pangaea_tab(CARBOSOL_PROFILE)
horizontes = _read_pangaea_tab(CARBOSOL_HORIZONS)

logging.info(f"Profiles shape:  {perfiles.shape}")
logging.info(f"Horizons shape:  {horizontes.shape}")

# 2) Resolver IDs y coords
perfiles, horizontes = _resolve_ids(perfiles, horizontes)

# 3) Target: cultivo desde Description
perfiles = _ensure_cultivo_from_description(perfiles)

sin_cultivo = perfiles["cultivo"].astype(str).str.strip().replace({"": np.nan}).isna().sum()
(OUTPUT_DIR / "cultivo_missing.txt").write_text(
    f"Perfiles SIN cultivo (vacío/NA): {sin_cultivo} de {len(perfiles)}\n", encoding="utf-8"
)

# 4) Agrupación avanzada
perfiles["cultivo_grupo"] = perfiles["cultivo"].astype(str).apply(assign_cultivo_group)

(perfiles["cultivo"].astype(str).fillna("NA").value_counts(dropna=False)
 .rename_axis("cultivo").reset_index(name="n")
 .to_csv(OUTPUT_DIR / "cultivo_frecuencias.csv", index=False))

(perfiles["cultivo_grupo"].astype(str).fillna("NA").value_counts(dropna=False)
 .rename_axis("cultivo_grupo").reset_index(name="n")
 .to_csv(OUTPUT_DIR / "cultivo_grupo_frecuencias.csv", index=False))

# 5) Estaciones AEMET + asignación
stations = get_aemet_stations()
assign = assign_nearest_station(perfiles, stations, year=2017)
perfiles = perfiles.merge(assign, on="profile_id", how="left")

# 6) Clima 2017: leer de CSV si existe; si no, descargar
if CLIMATE_2017_STATION_CSV.exists() and CLIMATE_2017_PROFILE_CSV.exists():
    logging.info("Leyendo clima 2017 desde CSV existente")
    clima_station = pd.read_csv(CLIMATE_2017_STATION_CSV)
    climate_df = pd.read_csv(CLIMATE_2017_PROFILE_CSV)
else:
    logging.info("Descargando clima 2017… (puede tardar)")
    clima_station, climate_df = build_and_save_climate_2017(perfiles)

# 7) Imputación por estación + proyección a perfil
if not clima_station.empty:
    clima_station_imp = impute_station_climate_2017(clima_station)
    clima_station_imp.to_csv(CLIMATE_2017_STATION_IMPUTED_CSV, index=False)

    mapping = perfiles[["profile_id","nearest_station"]].astype(str)
    climate_df_imp = mapping.merge(clima_station_imp, on="nearest_station", how="left")
    climate_df_imp.to_csv(CLIMATE_2017_PROFILE_IMPUTED_CSV, index=False)
else:
    climate_df_imp = pd.DataFrame(columns=["profile_id"])

# 8) Agregados climáticos por perfil
if not climate_df_imp.empty and "fecha" in climate_df_imp.columns:
    climate_df_imp["fecha"] = pd.to_datetime(climate_df_imp["fecha"], errors="coerce", format="mixed")
for c in ["tmed","tmax","tmin","prec","tpr"]:
    if c in climate_df_imp.columns:
        climate_df_imp[c] = pd.to_numeric(climate_df_imp[c], errors="coerce")

climate_df_imp["profile_id"] = climate_df_imp["profile_id"].astype(str) if "profile_id" in climate_df_imp.columns else climate_df_imp.get("profile_id")
climate_agg = build_climate_aggregates(climate_df_imp)

# 9) Horizons → profile
hz_summary = summarize_horizons_to_profile(horizontes)

# 10) Dataset final
perfiles_all = perfiles.copy()
perfiles_all["profile_id"] = perfiles_all["profile_id"].astype(str)
if not hz_summary.empty:
    hz_summary["profile_id"] = hz_summary["profile_id"].astype(str)
if not climate_agg.empty:
    climate_agg["profile_id"] = climate_agg["profile_id"].astype(str)

final_df = (perfiles_all
            .merge(hz_summary, on="profile_id", how="left")
            .merge(climate_agg, on="profile_id", how="left"))

final_path = OUTPUT_DIR / "dataset_final_2017_full.csv"
final_df.to_csv(final_path, index=False)
logging.info(f"Guardado: {final_path}")

# 11) Dataset para modelado (sin Otros)
final_df_model = final_df[final_df["cultivo_grupo"] != "Otros"].copy()
model_path = OUTPUT_DIR / "model" / "dataset_final_2017_model.csv"
safe_to_csv(final_df_model, model_path, index=False, encoding="utf-8-sig")
logging.info(f"Guardado: {model_path}")

final_df.head()

"""
## Sanity checks (rápidos)

- Distribución de clases `cultivo_grupo`
- % missing de clima agregado
- Ejemplos de perfiles con estación asignada
"""

# === Sanity checks ===

# 1) Distribución de clases
print("Distribución cultivo_grupo (FULL):")
display(
    final_df["cultivo_grupo"]
    .fillna("NA")
    .value_counts(dropna=False)
    .to_frame("n")
    .assign(pct=lambda d: (100*d["n"]/d["n"].sum()).round(2))
)

print("\nDistribución cultivo_grupo (MODEL, sin 'Otros'):")
display(
    final_df_model["cultivo_grupo"]
    .fillna("NA")
    .value_counts(dropna=False)
    .to_frame("n")
    .assign(pct=lambda d: (100*d["n"]/d["n"].sum()).round(2))
)

# 2) % missing de clima agregado
clima_cols = [c for c in final_df.columns if c.endswith("_2017") or c == "n_dias_lluvia"]
if clima_cols:
    miss_clima = (final_df[clima_cols].isna().mean().sort_values(ascending=False)*100).round(2)
    print("\n% missing en clima agregado (FULL):")
    display(miss_clima.to_frame("missing_pct"))
else:
    print("\nNo se han detectado columnas de clima agregado (*_2017 / n_dias_lluvia).")

# 3) Ejemplos de perfiles con estación asignada
cols_demo = [c for c in ["profile_id","lat","lon","cultivo","cultivo_grupo","nearest_station",
                         "station_name","station_provincia","distance_km","station_active_2017",
                         "station_rank_used"] if c in final_df.columns]
print("\nEjemplos de perfiles con estación asignada:")
display(final_df[cols_demo].dropna(subset=["nearest_station"]).sample(min(10, len(final_df)), random_state=42))

