import numpy as np
import cv2
import config
from pathlib import Path
from typing import Tuple, Dict, List


# Načítá všechna data KITTI pro jeden snímek: obraz, LiDAR, kalibraci a GT. Také provádí projekci LiDAR → kamera.
class KITTIDataLoader:

    # Sestaví absolutní cesty k podadresářům KITTI podle standardní struktury datasetu.
    def __init__(self, data_path: str = "data/kitti/training"):
        self.data_path = Path(data_path)
        self.image_path = self.data_path / "data_object_image_2" / "training" / "image_2"
        self.lidar_path = self.data_path / "data_object_velodyne" / "training" / "velodyne"
        self.calib_path = self.data_path / "data_object_calib" / "training" / "calib"
        self.label_path = self.data_path / "data_object_label_2" / "training" / "label_2" # GT anotace (label_2) jsou volitelné — pokud složka neexistuje, atribut has_labels = False a hodnocení přesnosti se automaticky přeskočí.

        # sorted() zajistí pořadí: 000000, 000001, 000002 atd
        self.image_files = sorted(list(self.image_path.glob("*.png")))
        self.lidar_files = sorted(list(self.lidar_path.glob("*.bin")))
        self.calib_files = sorted(list(self.calib_path.glob("*.txt")))
        self.label_files = sorted(list(self.label_path.glob("*.txt"))) if self.label_path.exists() else [] # GT - pokud složka neexistuje, vrátíme prázdný seznam

        print(f"Nalezeno {len(self.image_files)} obrázků")
        print(f"Nalezeno {len(self.lidar_files)} mračen bodů")
        print(f"Nalezeno {len(self.calib_files)} kalibračních souborů")
        if self.label_files:
            print(f"Nalezeno {len(self.label_files)} anotačních souborů (ground truth)")
        else:
            print("Anotační soubory (label_2) nenalezeny — hodnocení GT bude přeskočeno")

        # True = GT soubory nalezeny, False = hodnocení přesnosti neprobíhá
        self.has_labels = bool(self.label_files)

    # Načte PNG snímek jako pole pixelů (výška × šířka × 3 barvy).
    def load_image(self, index: int) -> np.ndarray:
        if index >= len(self.image_files):
            raise IndexError(f"Index {index} je mimo hranice") # Vyvolá IndexError pokud index překročí počet dostupných souborů.
        return cv2.imread(str(self.image_files[index]))

    # Načte binární soubor LiDAR jako tabulku bodů [x, y, z, intenzita].
    def load_lidar(self, index: int) -> np.ndarray:
        if index >= len(self.lidar_files):
            raise IndexError(f"Index {index} je mimo hranice")
        points = np.fromfile(str(self.lidar_files[index]), dtype=np.float32) # np.fromfile čte surová čísla float32.
        return points.reshape(-1, 4) # reshape(-1, 4) automaticky určí počet řádků N podle velikosti souboru.

    # Načte kalibrační matice ze souboru TXT = slovník numpy matic.
    #   P0–P3        : projekční matice kamer (3×4)
    #   R0_rect      : rektifikační rotační matice (3×3)
    #   Tr_velo_to_cam: transformace LiDAR → kamera (3×4)
    def load_calibration(self, index: int) -> Dict[str, np.ndarray]:
        if index >= len(self.calib_files):
            raise IndexError(f"Index {index} je mimo hranice")
        calib = {}
        with open(self.calib_files[index], 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()  # A strip() odstraní neviditelné znaky \n aby klíče šly najít ve slovníku.
                    values = [float(x) for x in value.strip().split()]
                    if len(values) == 12:
                        calib[key] = np.array(values).reshape(3, 4) # 12 čísel = matice 3×4.
                    elif len(values) == 9:
                        calib[key] = np.array(values).reshape(3, 3) # 9 čísel = matice 3×3.
        return calib

    # Načte GT anotace (správné rámečky objektů) ze souboru TXT.
    # Filtruje: pouze třídy z config, objekty příliš oříznuté nebo zakryté přeskočí.
    def load_labels(self, index: int, valid_classes: List[str] = None) -> List[Dict]:
        if not self.has_labels or index >= len(self.label_files): # Pokud GT soubory chybí nebo je index mimo rozsah -> prázdný seznam.
            return []

        if valid_classes is None: # Výchozí seznam tříd se načítá z config.py (Car / Pedestrian / Cyclist).
            valid_classes = config.GT_CLASSES

        labels = []
        with open(self.label_files[index], 'r') as f:
            for line in f:
                parts = line.strip().split() # Rozdělí řádek na sloupce podle mezer.
                if len(parts) < 15: # KITTI label má 15 sloupců - kratší řádek = neplatný.
                    continue
                obj_class  = parts[0] # Třída objektu z KITTI Datasetu.
                truncated  = float(parts[1]) # Míra oříznutí okrajem snímku [0.0–1.0].
                occluded   = int(parts[2]) # Míra zakrytí: 0 = plně vidět, 3 = neznámé
                left, top, right, bottom = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]) # 2D bounding box v pixelech: [4]levý, [5]horní, [6]pravý, [7]dolní okraj.
                loc_x, loc_y, loc_z      = float(parts[11]), float(parts[12]), float(parts[13]) # [11–13] 3D poloha objektu v souřadnicích kamery [m].
                distance = float(np.sqrt(loc_x**2 + loc_y**2 + loc_z**2)) # Vzdálenost objektu od senzoru - Pythagorova věta ve 3D: d = √(x² + y² + z²)  [m] + euklidovská norma 3D

                # Ignorujeme třídy mimo náš zájem.
                if obj_class not in valid_classes:
                    continue
                # Příliš oříznutý (>80%) nebo zakrytý objekt -> nespolehlivá GT -> přeskočit.
                if truncated > 0.8 or occluded > 2:
                    continue

                labels.append({
                    'class':      obj_class, # Název třídy pro per-class metriky.
                    'bbox':       [int(left), int(top), int(right), int(bottom)], # 2D box pro IoU.
                    'truncated':  truncated,
                    'occluded':   occluded,
                    'distance_m': distance,
                })
        return labels

    # Převede 3D body LiDAR na 2D pixely obrazu kamery ve třech krocích:
    # 1. Tr_velo_to_cam - přesun do souřadnicového systému kamery;
    # 2. R0_rect        - vyrovnání os kamer;
    # 3. P2             - projekce na rovinu obrazu (perspektiva, dělení Z).
    # Vrací pixelové souřadnice a masku viditelných bodů (před kamerou, v hranicích obrazu).
    def project_lidar_to_image(self, points: np.ndarray,
                               calib: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        P2             = calib.get('P2',             calib.get('P2:', None))
        R0_rect        = calib.get('R0_rect',        calib.get('R0_rect:', None))
        Tr_velo_to_cam = calib.get('Tr_velo_to_cam', calib.get('Tr_velo_to_cam:', None))

        if P2 is None or R0_rect is None or Tr_velo_to_cam is None:
            raise ValueError("Nebyly nalezeny potřebné kalibrační parametry")

        points_3d = points[:, :3]
        # Přidáme 1 na konec každého bodu - umožní provést otočení i posun jedním násobením, [X, Y, Z] -> [X, Y, Z, 1].
        points_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        points_cam = (Tr_velo_to_cam @ points_homo.T).T # krok 1.
        points_cam_rect = (R0_rect @ points_cam.T).T # krok 2.
        points_cam_rect_homo = np.hstack([points_cam_rect, np.ones((points_cam_rect.shape[0], 1))])
        points_img = (P2 @ points_cam_rect_homo.T).T # krok 3.
        points_2d  = points_img[:, :2] / points_img[:, 2:3] # Dělení Z vytvoří perspektivu - vzdálené objekty jsou menší.

        # Maska: True pouze pokud bod je před kamerou a uvnitř hranice obrazu.
        mask = (
            (points_img[:, 2] > 0) &
            (points_2d[:, 0] >= 0) &
            (points_2d[:, 0] < config.IMAGE_WIDTH) &
            (points_2d[:, 1] >= 0) &
            (points_2d[:, 1] < config.IMAGE_HEIGHT)
        )
        return points_2d.astype(int), mask

    # Pohodlná zkratka - načte obraz, LiDAR a kalibraci najednou.
    def get_sample(self, index: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        return self.load_image(index), self.load_lidar(index), self.load_calibration(index)
