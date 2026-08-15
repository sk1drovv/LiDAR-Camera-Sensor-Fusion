####################################################################################################################################################################
=======================================================
   SENZOROVÁ FÚZE LiDAR A KAMERY (KITTI Dataset)
=======================================================
   Bakalářská práce - Analýza metod senzorové fúze LiDARu a kamerového obrazu pro interpretaci scén.
   Autor:    Viacheslav Zlobin.
   Vedoucí:  Ing. Tomáš Klein, Ph.D.
   VŠB-TUO, FEI — Automotive Electronic Systems
   2026

======================================================
## Požadavky
======================================================

------------------------------------------------------
### Hardware
------------------------------------------------------

- CPU: libovolný 64bitový procesor (Intel/AMD)
- RAM: minimálně 8 GB (doporučeno 16 GB pro velká mračna bodů)
- Disk: ~50 GB volného místa (KITTI dataset)
- GPU: NVIDIA s podporou CUDA (volitelné - výrazně zrychluje YOLOv8)

------------------------------------------------------
### Software
------------------------------------------------------

- Python 3.8 nebo novější
- pip (správce balíčků Pythonu)
- NVIDIA CUDA Toolkit 11.x nebo 12.x (pouze pro GPU akceleraci)

------------------------------------------------------
### Závislosti Pythonu
------------------------------------------------------

Hlavní knihovny použité v projektu:

|------------------------------------------------------------------------------|
|    Knihovna    |   Verze    |                     Účel                       |
|----------------|------------|------------------------------------------------|
|     numpy      | >= 1.21.0  |     numerické výpočty, práce s mračny bodů     |
|----------------|------------|------------------------------------------------|
|  opencv-python |  >= 4.5.0  |        načítání obrazů, kreslení bboxů         |
|----------------|------------|------------------------------------------------|
|   matplotlib   |  >= 3.5.0  |          vizualizace, srovnávací grafy         |
|----------------|------------|------------------------------------------------|
|     pandas     |  >= 1.3.0  |         export statistik a metrik do CSV       |
|----------------|------------|------------------------------------------------|
|     torch      |  >= 1.9.0  |         hluboké učení (backend YOLOv8)         |
|----------------|------------|------------------------------------------------|
|   ultralytics  |  >= 8.0.0  |        YOLOv8 detekce objektů v obraze         |
|----------------|------------|------------------------------------------------|
|    hdbscan     |  >= 0.8.0  |       hierarchické shlukování bodů LiDAR       |
|------------------------------------------------------------------------------|

Kompletní seznam včetně závislostí druhé úrovně je v souboru "requirements.txt".

======================================================
## Instalace
======================================================

------------------------------------------------------
### 1. Příprava projektu
------------------------------------------------------

Rozbalte zdrojové soubory projektu do libovolné složky, např.:

    D:\Python\sensor_fusion_kitti\

V této složce by měly být soubory: "main.py", "config.py", "data_loader.py",
"fusion_methods.py", "visualization.py", "analysis.py", "requirements.txt" a další.

------------------------------------------------------
### 2. Vytvoření virtuálního prostředí (doporučeno)
------------------------------------------------------

Otevřete příkazový řádek ve složce projektu:

    # Windows
    python -m venv .venv
    .venv\Scripts\activate

------------------------------------------------------
### 3. Instalace závislostí
------------------------------------------------------

    pip install -r requirements.txt

------------------------------------------------------
### 4. Stažení vah YOLOv8
------------------------------------------------------

Soubor "yolov8n.pt" se automaticky stáhne při prvním spuštění programu (cca 6 MB).
Alternativně lze stáhnout ručně:

    yolo download model=yolov8n.pt

------------------------------------------------------
### 5. Příprava datasetu KITTI
------------------------------------------------------

Stáhněte KITTI Object Detection Dataset z oficiálních stránek:
http://www.cvlibs.net/datasets/kitti/eval_object.php

Potřebné archivy:
- "data_object_image_2.zip"     (~12 GB)  - snímky levé kamery
- "data_object_velodyne.zip"    (~29 GB)  - mračna bodů LiDAR
- "data_object_calib.zip"       (~16 MB)  - kalibrační soubory
- "data_object_label_2.zip"     (~5 MB)   - GT anotace

Rozbalte je do složky "DATASET/" v kořeni projektu tak,
aby výsledná struktura odpovídala:

    DATASET/
    |-- data_object_image_2/training/image_2/*.png
    |-- data_object_velodyne/training/velodyne/*.bin
    |-- data_object_calib/training/calib/*.txt
    +-- data_object_label_2/training/label_2/*.txt

V config.py je nastavena cesta "../Sample_data/DATASET" - odkazuje na přiložený mini-dataset 10 snímků pro demonstraci. Pro vlastní plný KITTI dataset upravte KITTI_DATA_PATH a KITTI_TRAINING_PATH na cestu k vaší kopii KITTI.

------------------------------------------------------
### 6. Ověření instalace
------------------------------------------------------

    python main.py

Při dotazu na počet snímků zadejte "5" pro rychlý test.
Po dokončení by se ve složce "results/" měly objevit:
- "result_frame_000000.png" … "result_frame_000004.png"
- "statistics.csv", "metrics.csv", "report.txt"

------------------------------------------------------
### Volitelné: GPU akcelerace (CUDA)
------------------------------------------------------

Pro výrazné zrychlení detekce YOLOv8 lze využít NVIDIA GPU s podporou CUDA.
PyTorch s podporou CUDA se instaluje samostatně podle verze ovladače:

    # příklad pro CUDA 12.1
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Detekce GPU probíhá automaticky — pokud CUDA není dostupná,
program se vrátí na CPU bez nutnosti zásahu.

------------------------------------------------------
### Řešení častých problémů
------------------------------------------------------

|----------------------------------------------------------------------------------------------------------------|
|                   *Chyba*                      |                           *Řešení*                            |
|------------------------------------------------|---------------------------------------------------------------|
|        "ModuleNotFoundError: hdbscan"          |                    "pip install hdbscan"                      |
|------------------------------------------------|---------------------------------------------------------------|
|     "CHYBA: Data KITTI nebyla nalezena"        |      Zkontrolovat cestu "KITTI_DATA_PATH" v "config.py"       |
|------------------------------------------------|---------------------------------------------------------------|
|    "Anotační soubory (label_2) nenalezeny"     |   Stáhnout "data_object_label_2.zip" — hodnocení vyžaduje GT  |
|------------------------------------------------|---------------------------------------------------------------|
|              "CUDA out of memory"              |      Nastavit "device = 'cpu'" nebo zmenšit batch v YOLO      |
|----------------------------------------------------------------------------------------------------------------|








======================================================
## Architektura systému
======================================================

Pipeline zpracování každého snímku:

    Načtení dat (snímek + LiDAR + kalibrace)
            |
            |-- Projekce LiDAR -> kamera (Tr_velo_to_cam -> R0_rect -> P2)
            |
            |-- Detekce kamerou -- YOLOv8 na celém snímku
            |
            |-- Detekce LiDARem -- Odstranění vozovky -> HDBSCAN shlukování -> NMS
            |
            |-- Decision-Level Fusion -- Nezávislé detekce -> párování přes IoU + středy -> sloučení
            |
            |-- Feature-Level Fusion -- LiDAR shluky -> výřez ROI -> YOLOv8 na výřezech
            |
            |-- Vyhodnocení přesnosti vs. Ground Truth (Precision / Recall / F1)
            |
            +-- Vizualizace (6-panelový srovnávací graf)

======================================================
## Metody detekce
======================================================

------------------------------------------------------
### 1. Kamera (YOLOv8)
------------------------------------------------------
- Model: YOLOv8n (nano) - yolov8n.pt předtrénované váhy (COCO, ~6 MB)  
- Práh confidence: 0.55 (pro minimalizaci FP)
- Třídy: Car, Pedestrian, Cyclist (mapování COCO -> KITTI)
- Podpora GPU (CUDA) s automatickým fallbackem na CPU

------------------------------------------------------
### 2. LiDAR (HDBSCAN)
------------------------------------------------------
- Odstranění bodů vozovky podle výšky Z (práh: -1.4 m)
- Voxelizace pro mračna > 50 000 bodů (velikost voxelu: 0.1 m)
- Shlukování HDBSCAN s adaptivním epsilon na základě průměrné vzdálenosti
- Geometrický filtr shluků: výška 0.5-3.5 m, šířka 0.4-3.0 m, hloubka 0.5-5.0 m
- Filtr hustoty bodů (> 5 bodů/m3)
- Non-Maximum Suppression (NMS) pro odstranění duplicit

------------------------------------------------------
### 3. Decision-Level Fusion
------------------------------------------------------
- Kamera a LiDAR detekují nezávisle
- Hladové (greedy) párování: kamerové detekce (sestupně podle confidence) s LiDAR detekcemi
- Skóre párování: 0.6 * IoU + 0.4 (1 - dist_centers / 150)
- Práh párování: 0.20
- Confidence výsledku: 0.7 * kamera + 0.3 * LiDAR
- Tři typy výstupů: fused / image_only / lidar_only

------------------------------------------------------
### 4. Feature-Level Fusion
------------------------------------------------------
- LiDAR určuje KDE hledat (ROI ze shluků + okraj 60 px)
- YOLOv8 klasifikuje CO se v oblasti nachází
- Práh confidence: 0.25 (nižší kvůli menšímu kontextu výřezu)
- Malé výřezy (< 128 px) jsou zvětšeny nahoru
- Jeden shluk = jeden objekt (bere se nejlepší detekce)

======================================================
## Struktura projektu
======================================================

    sensor_fusion_kitti/
    |
    |-- main.py                    # Hlavní pipeline: načtení -> detekce -> fúze -> vyhodnocení -> vizualizace
    |-- config.py                  # Konfigurace: cesty, parametry, přepínače metod
    |-- data_loader.py             # Načítač dat KITTI (snímky, LiDAR, kalibrace, GT)
    |-- fusion_methods.py          # Implementace metod fúze (Decision + Feature)
    |-- visualization.py           # Vizualizace: 6-panelové srovnávací grafy
    |-- analysis.py                # Vyhodnocení přesnosti: TP/FP/FN, Precision/Recall/F1 podle tříd a vzdáleností
    |-- sensitivity_analysis.py    # Analýza citlivosti parametrů Decision Fusion
    |-- analyze_gt_sizes.py        # Validace geometrického filtru podle skutečných rozměrů GT objektů
    |-- yolov8n.pt                 # Předtrénované váhy YOLOv8 nano (COCO, ~6 MB)  
    |-- requirements.txt           # Závislosti Pythonu
    |-- __init__.py                # Verze balíčku (1.0.0)
    |
    |-- DATASET/                                          # Dataset KITTI (standardní struktura)
    |   |-- data_object_image_2/training/image_2/         # PNG snímky kamery
    |   |-- data_object_velodyne/training/velodyne/       # Binární soubory LiDAR (.bin)
    |   |-- data_object_calib/training/calib/             # Kalibrační matice (.txt)
    |   +-- data_object_label_2/training/label_2/         # GT anotace (.txt)
    |
    |-- results/                       # Výstupní soubory
    |   |-- result_frame_XXXXXX.png    # 6-panelové vizualizace
    |   |-- statistics.csv             # Statistika po snímcích
    |   |-- metrics.csv                # Metriky P/R/F1 (metoda x třída x vzdálenost)
    |   +-- report.txt                 # Textová zpráva
    |
    |-- sensitivity_results.csv        # Výsledky analýzy citlivosti

======================================================
## Příprava dat
======================================================

Projekt využívá KITTI Object Detection Dataset. Data je nutné stáhnout a umístit do složky DATASET/ s následující strukturou:

1. data_object_image_2 - barevné snímky levé kamery (.png)
2. data_object_velodyne - mračna bodů Velodyne LiDAR (.bin, formát float32: x, y, z, intenzita)
3. data_object_calib - kalibrační soubory (projekční matice P0-P3, R0_rect, Tr_velo_to_cam)
4. data_object_label_2 - GT anotace (volitelné, pro vyhodnocení přesnosti)

Každý soubor má 6-místný index (např. 000043.png, 000043.bin, 000043.txt).

------------------------------------------------------
### Projekce LiDAR -> kamera
------------------------------------------------------
Tříkroková transformace 3D souřadnic LiDARu do pixelových souřadnic:

1. Tr_velo_to_cam - převod do souřadnicového systému kamery (3x4)
2. R0_rect - rektifikace (vyrovnání os kamer, 3x3)
3. P2 - perspektivní projekce na rovinu snímku (3x4, dělení hodnotou Z)

Maska viditelnosti: bod je považován za viditelný, pokud Z > 0 a souřadnice leží uvnitř hranic snímku (1242 x 375).

======================================================
## Konfigurace
======================================================

Soubor config.py obsahuje všechny parametry systému:

|----------------------------------------------------------------------------------------------------|
| 	*Parametr*       |         *Hodnota*        |                    *Popis*                     |
|------------------------|--------------------------|------------------------------------------------|
|     KITTI_DATA_PATH    |         "DATASET"        |            Kořenová složka datasetu            |
|------------------------|--------------------------|------------------------------------------------|
|    IMAGE_WIDTH/HEIGHT  |        1242 x 375        |             Rozlišení snímků KITTI             |
|------------------------|--------------------------|------------------------------------------------|
|        VISUALIZE       |           True           |          Vytváření 6-panelových grafů          |
|------------------------|--------------------------|------------------------------------------------|
|      SAVE_RESULTS      |           True           |            Ukládání výsledků na disk           |
|------------------------|--------------------------|------------------------------------------------|
|  USE_CAMERA_DETECTION  |           True           |            Aktivace detekce kamerou            |
|------------------------|--------------------------|------------------------------------------------|
|   USE_LIDAR_DETECTION  |           True           |            Aktivace detekce LiDARem            |
|------------------------|--------------------------|------------------------------------------------|
|   USE_DECISION_FUSION  |           True           |          Aktivace Decision-Level Fusion        |
|------------------------|--------------------------|------------------------------------------------|
|    USE_FEATURE_FUSION  |           True           |          Aktivace Feature-Level Fusion         |
|------------------------|--------------------------|------------------------------------------------|
|     EVALUATE_WITH_GT   |           True           |          Vyhodnocení vůči ground truth         |
|------------------------|--------------------------|------------------------------------------------|
|     GT_IOU_THRESHOLD   |            0.5           |            Práh IoU pro párování s GT          |
|------------------------|--------------------------|------------------------------------------------|
|       GT_CLASSES       | Car, Pedestrian, Cyclist |                 Sledované třídy                |
|------------------------|--------------------------|------------------------------------------------|
|      DISTANCE_BINS     |     [0, 20, 40, 80]      |              Vzdálenostní pásma (m)            |
|----------------------------------------------------------------------------------------------------|


======================================================
## Spuštění
======================================================

------------------------------------------------------
### Hlavní pipeline
------------------------------------------------------

    python main.py

Program si vyžádá počet snímků ke zpracování (od 1 do maximálního počtu dostupných). Výstupem je konzolový výpis s ukazatelem průběhu, tabulky přesnosti a soubory ve složce results/.

------------------------------------------------------
### Analýza citlivosti
------------------------------------------------------
    python sensitivity_analysis.py

Ověřuje, zda baseline hodnoty konstant v fusion_methods.py jsou optimální. Spustí pipeline na 1500 snímcích pro každou variantu parametru a porovná F1 s baseline.

Zkoumané skupiny parametrů:
|------------------------------------------------------------------------------------|
|  *Skupina*  |           *Parametr*            |         *Zkoumané hodnoty*         |
|-------------|---------------------------------|------------------------------------|
|      A      | Váhy match_score (IoU / středy) |     0.5/0.5, 0.6/0.4, 0.7/0.3      |
|-------------|---------------------------------|------------------------------------|
|      B      | Práh párování                   |    0.10, 0.15, 0.20, 0.25, 0.30    |  
|-------------|---------------------------------|------------------------------------| 
|      C      | Normalizace vzdálenosti středů  |          100, 150, 200 px          | 
|-------------|---------------------------------|------------------------------------|        
|      D      | Váhy confidence (kamera / LiDAR)| 0.5/0.5, 0.6/0.4, 0.7/0.3, 0.8/0.2 |
|-------------|---------------------------------|------------------------------------|
|      E      | Práh odstranění vozovky         |         -1.2, -1.4, -1.6 m         |        
|------------------------------------------------------------------------------------|

Výsledky se ukládají do sensitivity_results.csv.
Výstup:
- konzolový výpis F1/P/R + souhrnná tabulka odchylek od baseline
- soubor sensitivity_results.csv

Doba běhu: ~15-30 min na GPU. Pro rychlý test lze v hlavičce skriptu snížit N_FRAMES na 100.

------------------------------------------------------
### Validace geometrického filtru
------------------------------------------------------
Měří skutečné 3D rozměry GT objektů v KITTI a ověřuje, jaký podíl objektů splňuje meze filtru filter_cluster() (h: 0.5-3.5 m, w: 0.4-3.0 m, d: 0.5-5.0 m).

    python analyze_gt_sizes.py

Výstup (pouze konzole):
- statistika rozměrů pro každou třídu (min/max/mean, p5/p95)
- procento objektů splňujících všechny meze filtru

Doba běhu: ~5-10 s (bez GPU).

======================================================
## Výstupní data
======================================================

------------------------------------------------------
### 6-panelová vizualizace (result_frame_XXXXXX.png)
------------------------------------------------------

Každý zpracovaný snímek generuje obrázek ze 6 panelů:

|--------------------------------------------------------------------------------------------------------------------|
|            |                *Sloupec 1*               |          *Sloupec 2*           |        *Sloupec 3*        |
|------------|------------------------------------------|--------------------------------|---------------------------|
|  *Řada 1*  | LiDAR projekce (barva podle vzdálenosti) | Kamera YOLOv8 (zelené rámečky) | Feature Fusion (oranžové) |
|------------|------------------------------------------|--------------------------------|---------------------------|
|  *Řada 2*  |            LiDAR shluky (modré)          |    Decision Fusion (červené)   |      Porovnání metod      |
|--------------------------------------------------------------------------------------------------------------------|


------------------------------------------------------
### statistics.csv
------------------------------------------------------

Statistika po snímcích: počet bodů LiDAR, viditelných bodů, detekcí každé metody, doba zpracování.


------------------------------------------------------
### metrics.csv
------------------------------------------------------

Úplná tabulka metrik (Precision / Recall / F1 / TP / FP / FN) pro každou kombinaci: metoda x třída x vzdálenostní pásmo.


------------------------------------------------------
### report.txt
------------------------------------------------------

Textová zpráva: aktivní metody, statistika LiDAR, průměrné detekce, FPS, přesnost vůči GT s rozdělením podle tříd a vzdáleností.

======================================================
## Metriky a vyhodnocení
======================================================

Vyhodnocení se provádí hladovým (greedy) párováním detekcí s GT anotacemi (od nejvyšší confidence směrem dolů). Detekce je považována za TP při IoU >= 0.5 s dosud nespárovaným GT objektem. Metriky jsou agregovány:

- Podle tříd: Car, Pedestrian, Cyclist, ALL
- Podle vzdálenosti: 0-20 m, 20-40 m, 40-80 m, > 80 m, ALL
- Podle metod: camera, lidar, decision_fusion, feature_fusion

GT anotace s truncation > 0.8 nebo occlusion > 2 jsou automaticky vyloučeny jako nespolehlivé.

####################################################################################################################################################################
