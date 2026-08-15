🇨🇿 Česky | [🇬🇧 English](README.md)
# Senzorová fúze LiDARu a kamery (KITTI Dataset)

**Bakalářská práce** - Analýza metod senzorové fúze LiDARu a kamerového obrazu pro interpretaci scén

**Autor:** Viacheslav Zlobin
**Vedoucí:** Ing. Tomáš Klein, Ph.D.
**VŠB-TUO, FEI** - Automobilové elektronické systémy, 2026

---

## Zadání

1. Proveďte rešerši současných metod fúze LiDARu a kamerového obrazu zaměřených na interpretaci scén, například pro detekci objektů, segmentaci nebo porozumění prostředí.
2. Popište a analyzujte přístupy k fúzi dat na úrovni vlastností (feature-level) a na úrovni rozhodnutí (decision-level), včetně jejich výhod, nevýhod a vhodného použití.
3. Na poskytnutých nahrávkách ukažte princip fungování vybraných metod fúze a vizuálně nebo analyticky porovnejte jejich výsledky.
4. Vyhodnoťte přesnost, robustnost a vhodnost jednotlivých metod fúze pro interpretaci scén.

---

## Požadavky

### Hardware

- **CPU:** libovolný 64bitový procesor (Intel/AMD)
- **RAM:** minimálně 8 GB (doporučeno 16 GB pro velká mračna bodů)
- **Disk:** ~50 GB volného místa (KITTI dataset)
- **GPU:** NVIDIA s podporou CUDA (volitelné — výrazně zrychluje YOLOv8)

### Software

- Python 3.8 nebo novější
- pip (správce balíčků Pythonu)
- NVIDIA CUDA Toolkit 11.x nebo 12.x (pouze pro GPU akceleraci)

### Závislosti Pythonu

| Knihovna | Verze | Účel |
|---|---|---|
| numpy | >= 1.21.0 | numerické výpočty, práce s mračny bodů |
| opencv-python | >= 4.5.0 | načítání obrazů, kreslení bboxů |
| matplotlib | >= 3.5.0 | vizualizace, srovnávací grafy |
| pandas | >= 1.3.0 | export statistik a metrik do CSV |
| torch | >= 1.9.0 | hluboké učení (backend YOLOv8) |
| ultralytics | >= 8.0.0 | YOLOv8 detekce objektů v obraze |
| hdbscan | >= 0.8.0 | hierarchické shlukování bodů LiDAR |

Kompletní seznam včetně závislostí druhé úrovně je v souboru `requirements.txt`.

---

## Instalace

### 1. Příprava projektu

Rozbalte zdrojové soubory projektu do libovolné složky, např.:

```
D:\Python\sensor_fusion_kitti\
```

V této složce by měly být soubory `main.py`, `config.py`, `data_loader.py`, `fusion_methods.py`, `visualization.py`, `analysis.py`, `requirements.txt` a další.

### 2. Vytvoření virtuálního prostředí (doporučeno)

Otevřete příkazový řádek ve složce projektu:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalace závislostí

```bash
pip install -r requirements.txt
```

### 4. Stažení vah YOLOv8

Soubor `yolov8n.pt` se automaticky stáhne při prvním spuštění programu (cca 6 MB). Alternativně jej lze stáhnout ručně:

```bash
yolo download model=yolov8n.pt
```

### 5. Příprava datasetu KITTI

Stáhněte KITTI Object Detection Dataset z oficiálních stránek:
http://www.cvlibs.net/datasets/kitti/eval_object.php

Potřebné archivy:

| Archiv | Velikost | Obsah |
|---|---|---|
| `data_object_image_2.zip` | ~12 GB | snímky levé kamery |
| `data_object_velodyne.zip` | ~29 GB | mračna bodů LiDAR |
| `data_object_calib.zip` | ~16 MB | kalibrační soubory |
| `data_object_label_2.zip` | ~5 MB | GT anotace |

Rozbalte je do složky `DATASET/` v kořeni projektu tak, aby výsledná struktura odpovídala:

```
DATASET/
├── data_object_image_2/training/image_2/*.png
├── data_object_velodyne/training/velodyne/*.bin
├── data_object_calib/training/calib/*.txt
└── data_object_label_2/training/label_2/*.txt
```

V `config.py` je nastavena cesta `../Sample_data/DATASET` — odkazuje na přiložený mini-dataset 10 snímků pro demonstraci. Pro vlastní plný KITTI dataset upravte `KITTI_DATA_PATH` a `KITTI_TRAINING_PATH` na cestu k vaší kopii KITTI.

### 6. Ověření instalace

```bash
python main.py
```

Při dotazu na počet snímků zadejte `5` pro rychlý test. Po dokončení by se ve složce `results/` měly objevit:

- `result_frame_000000.png` … `result_frame_000004.png`
- `statistics.csv`, `metrics.csv`, `report.txt`

### Volitelné: GPU akcelerace (CUDA)

Pro výrazné zrychlení detekce YOLOv8 lze využít NVIDIA GPU s podporou CUDA. PyTorch s podporou CUDA se instaluje samostatně podle verze ovladače:

```bash
# příklad pro CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Detekce GPU probíhá automaticky — pokud CUDA není dostupná, program se vrátí na CPU bez nutnosti zásahu.

### Řešení častých problémů

| Chyba | Řešení |
|---|---|
| `ModuleNotFoundError: hdbscan` | `pip install hdbscan` |
| `CHYBA: Data KITTI nebyla nalezena` | zkontrolovat cestu `KITTI_DATA_PATH` v `config.py` |
| `Anotační soubory (label_2) nenalezeny` | stáhnout `data_object_label_2.zip` — hodnocení vyžaduje GT |
| `CUDA out of memory` | nastavit `device = 'cpu'` nebo zmenšit batch v YOLO |

---

## Architektura systému

Pipeline zpracování každého snímku:

```
Načtení dat (snímek + LiDAR + kalibrace)
        │
        ├── Projekce LiDAR → kamera (Tr_velo_to_cam → R0_rect → P2)
        │
        ├── Detekce kamerou — YOLOv8 na celém snímku
        │
        ├── Detekce LiDARem — odstranění vozovky → HDBSCAN shlukování → NMS
        │
        ├── Decision-level fúze — nezávislé detekce → párování přes IoU + středy → sloučení
        │
        ├── Feature-level fúze — LiDAR shluky → výřez ROI → YOLOv8 na výřezech
        │
        ├── Vyhodnocení přesnosti vs. ground truth (Precision / Recall / F1)
        │
        └── Vizualizace (6panelový srovnávací graf)
```

---

## Metody detekce

### 1. Kamera (YOLOv8)

- Model: YOLOv8n (nano) — `yolov8n.pt` předtrénované váhy (COCO, ~6 MB)
- Práh confidence: 0.55 (pro minimalizaci FP)
- Třídy: Car, Pedestrian, Cyclist (mapování COCO → KITTI)
- Podpora GPU (CUDA) s automatickým fallbackem na CPU

### 2. LiDAR (HDBSCAN)

- Odstranění bodů vozovky podle výšky Z (práh: −1,4 m)
- Voxelizace pro mračna nad 50 000 bodů (velikost voxelu: 0,1 m)
- Shlukování HDBSCAN s adaptivním epsilon na základě průměrné vzdálenosti
- Geometrický filtr shluků: výška 0,5–3,5 m, šířka 0,4–3,0 m, hloubka 0,5–5,0 m
- Filtr hustoty bodů (> 5 bodů/m³)
- Non-Maximum Suppression (NMS) pro odstranění duplicit

### 3. Decision-level fúze

- Kamera a LiDAR detekují nezávisle
- Hladové (greedy) párování: kamerové detekce (sestupně podle confidence) s LiDAR detekcemi
- Skóre párování: `0,6 × IoU + 0,4 × (1 − dist_centers / 150)`
- Práh párování: 0,20
- Confidence výsledku: `0,7 × kamera + 0,3 × LiDAR`
- Tři typy výstupů: `fused` / `image_only` / `lidar_only`

### 4. Feature-level fúze

- LiDAR určuje **kde** hledat (ROI ze shluků + okraj 60 px)
- YOLOv8 klasifikuje **co** se v oblasti nachází
- Práh confidence: 0,25 (nižší kvůli menšímu kontextu výřezu)
- Malé výřezy (< 128 px) jsou zvětšeny nahoru
- Jeden shluk = jeden objekt (bere se nejlepší detekce)

---

## Struktura projektu

```
sensor_fusion_kitti/
│
├── main.py                    # Hlavní pipeline: načtení → detekce → fúze → vyhodnocení → vizualizace
├── config.py                  # Konfigurace: cesty, parametry, přepínače metod
├── data_loader.py             # Načítač dat KITTI (snímky, LiDAR, kalibrace, GT)
├── fusion_methods.py          # Implementace metod fúze (decision + feature)
├── visualization.py           # Vizualizace: 6panelové srovnávací grafy
├── analysis.py                # Vyhodnocení přesnosti: TP/FP/FN, precision/recall/F1 podle tříd a vzdáleností
├── sensitivity_analysis.py    # Analýza citlivosti parametrů decision fúze
├── analyze_gt_sizes.py        # Validace geometrického filtru podle skutečných rozměrů GT objektů
├── yolov8n.pt                 # Předtrénované váhy YOLOv8 nano (COCO, ~6 MB)
├── requirements.txt           # Závislosti Pythonu
├── __init__.py                # Verze balíčku (1.0.0)
│
├── DATASET/                                          # Dataset KITTI (standardní struktura)
│   ├── data_object_image_2/training/image_2/         # PNG snímky kamery
│   ├── data_object_velodyne/training/velodyne/       # Binární soubory LiDAR (.bin)
│   ├── data_object_calib/training/calib/             # Kalibrační matice (.txt)
│   └── data_object_label_2/training/label_2/         # GT anotace (.txt)
│
├── results/                       # Výstupní soubory
│   ├── result_frame_XXXXXX.png    # 6panelové vizualizace
│   ├── statistics.csv             # Statistika po snímcích
│   ├── metrics.csv                # Metriky P/R/F1 (metoda × třída × vzdálenost)
│   └── report.txt                 # Textová zpráva
│
└── sensitivity_results.csv        # Výsledky analýzy citlivosti
```

---

## Příprava dat

Projekt využívá KITTI Object Detection Dataset. Data je nutné stáhnout a umístit do složky `DATASET/` s následující strukturou:

1. **data_object_image_2** — barevné snímky levé kamery (`.png`)
2. **data_object_velodyne** — mračna bodů Velodyne LiDAR (`.bin`, formát float32: x, y, z, intenzita)
3. **data_object_calib** — kalibrační soubory (projekční matice P0–P3, R0_rect, Tr_velo_to_cam)
4. **data_object_label_2** — GT anotace (volitelné, pro vyhodnocení přesnosti)

Každý soubor má 6místný index (např. `000043.png`, `000043.bin`, `000043.txt`).

### Projekce LiDAR → kamera

Tříkroková transformace 3D souřadnic LiDARu do pixelových souřadnic:

1. **Tr_velo_to_cam** — převod do souřadnicového systému kamery (3×4)
2. **R0_rect** — rektifikace (vyrovnání os kamer, 3×3)
3. **P2** — perspektivní projekce na rovinu snímku (3×4, dělení hodnotou Z)

Maska viditelnosti: bod je považován za viditelný, pokud Z > 0 a souřadnice leží uvnitř hranic snímku (1242 × 375).

---

## Konfigurace

Soubor `config.py` obsahuje všechny parametry systému:

| Parametr | Hodnota | Popis |
|---|---|---|
| `KITTI_DATA_PATH` | `"DATASET"` | kořenová složka datasetu |
| `IMAGE_WIDTH/HEIGHT` | 1242 × 375 | rozlišení snímků KITTI |
| `VISUALIZE` | `True` | vytváření 6panelových grafů |
| `SAVE_RESULTS` | `True` | ukládání výsledků na disk |
| `USE_CAMERA_DETECTION` | `True` | aktivace detekce kamerou |
| `USE_LIDAR_DETECTION` | `True` | aktivace detekce LiDARem |
| `USE_DECISION_FUSION` | `True` | aktivace decision-level fúze |
| `USE_FEATURE_FUSION` | `True` | aktivace feature-level fúze |
| `EVALUATE_WITH_GT` | `True` | vyhodnocení vůči ground truth |
| `GT_IOU_THRESHOLD` | 0,5 | práh IoU pro párování s GT |
| `GT_CLASSES` | Car, Pedestrian, Cyclist | sledované třídy |
| `DISTANCE_BINS` | `[0, 20, 40, 80]` | vzdálenostní pásma (m) |

---

## Spuštění

### Hlavní pipeline

```bash
python main.py
```

Program si vyžádá počet snímků ke zpracování (od 1 do maximálního počtu dostupných). Výstupem je konzolový výpis s ukazatelem průběhu, tabulky přesnosti a soubory ve složce `results/`.

### Analýza citlivosti

```bash
python sensitivity_analysis.py
```

Ověřuje, zda baseline hodnoty konstant v `fusion_methods.py` jsou optimální. Spustí pipeline na 1500 snímcích pro každou variantu parametru a porovná F1 s baseline.

Zkoumané skupiny parametrů:

| Skupina | Parametr | Zkoumané hodnoty |
|---|---|---|
| A | váhy `match_score` (IoU / středy) | 0,5/0,5, 0,6/0,4, 0,7/0,3 |
| B | práh párování | 0,10, 0,15, 0,20, 0,25, 0,30 |
| C | normalizace vzdálenosti středů | 100, 150, 200 px |
| D | váhy confidence (kamera / LiDAR) | 0,5/0,5, 0,6/0,4, 0,7/0,3, 0,8/0,2 |
| E | práh odstranění vozovky | −1,2, −1,4, −1,6 m |

Výstup:
- konzolový výpis F1/P/R a souhrnná tabulka odchylek od baseline
- soubor `sensitivity_results.csv`

Doba běhu: cca 15–30 min na GPU. Pro rychlý test lze v hlavičce skriptu snížit `N_FRAMES` na 100.

### Validace geometrického filtru

```bash
python analyze_gt_sizes.py
```

Měří skutečné 3D rozměry GT objektů v KITTI a ověřuje, jaký podíl objektů splňuje meze filtru `filter_cluster()` (h: 0,5–3,5 m, w: 0,4–3,0 m, d: 0,5–5,0 m).

Výstup (pouze konzole):
- statistika rozměrů pro každou třídu (min/max/mean, p5/p95)
- procento objektů splňujících všechny meze filtru

Doba běhu: cca 5–10 s (bez GPU).

---

## Výstupní data

### 6panelová vizualizace (`result_frame_XXXXXX.png`)

Každý zpracovaný snímek generuje obrázek ze 6 panelů:

| | Sloupec 1 | Sloupec 2 | Sloupec 3 |
|---|---|---|---|
| **Řada 1** | LiDAR projekce (barva podle vzdálenosti) | Kamera YOLOv8 (zelené rámečky) | Feature fúze (oranžové) |
| **Řada 2** | LiDAR shluky (modré) | Decision fúze (červené) | Porovnání metod |

### `statistics.csv`

Statistika po snímcích: počet bodů LiDAR, viditelných bodů, detekcí každé metody, doba zpracování.

### `metrics.csv`

Úplná tabulka metrik (precision / recall / F1 / TP / FP / FN) pro každou kombinaci metoda × třída × vzdálenostní pásmo.

### `report.txt`

Textová zpráva: aktivní metody, statistika LiDAR, průměrné detekce, FPS, přesnost vůči GT s rozdělením podle tříd a vzdáleností.

---

## Metriky a vyhodnocení

Vyhodnocení se provádí hladovým (greedy) párováním detekcí s GT anotacemi, od nejvyšší confidence směrem dolů. Detekce je považována za TP při IoU ≥ 0,5 s dosud nespárovaným GT objektem. Metriky jsou agregovány:

- **Podle tříd:** Car, Pedestrian, Cyclist, ALL
- **Podle vzdálenosti:** 0–20 m, 20–40 m, 40–80 m, > 80 m, ALL
- **Podle metod:** camera, lidar, decision_fusion, feature_fusion

GT anotace s truncation > 0,8 nebo occlusion > 2 jsou automaticky vyloučeny jako nespolehlivé.
