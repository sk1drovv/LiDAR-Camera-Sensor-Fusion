import numpy as np
import math
import cv2
from typing import List, Dict
import hdbscan
from ultralytics import YOLO
import torch

# Převodní tabulka COCO -> KITTI:
COCO_TO_KITTI = {
    0: 'Pedestrian',
    1: 'Cyclist',     # bicycle
    2: 'Car',         # car
    3: 'Cyclist',     # motorcycle
    7: 'Car',         # truck
}
# Seznam čísel tříd které sledujeme.
TARGET_CLASSES = list(COCO_TO_KITTI.keys())


# Odstraní body vozovky z mračna LiDAR.
def remove_ground(points: np.ndarray, z_threshold: float = -1.4) -> np.ndarray:
    return points[:, 2] > z_threshold

# -------------------------------------------------------
# HDBSCAN, shlukuje body LiDAR -> každý cluster = jeden objekt.
# -------------------------------------------------------
def cluster_lidar(points: np.ndarray) -> List[np.ndarray]:
    n_points = len(points)
    if n_points < 5: # Příliš málo bodů → nelze shlukovat
        return []

    xyz = points[:, :3]
    avg_dist = np.mean(np.sqrt(np.sum(xyz ** 2, axis=1)))
    eps = float(np.clip(0.6 + (avg_dist / 50.0) * 0.9, 0.6, 1.5)) # Adaptivní epsilon.

# Voxelizace
    use_voxel = n_points > 50_000
    if use_voxel:
        voxel_size = 0.1
        voxel_coords = np.floor(xyz / voxel_size).astype(np.int32)
        unique_voxels, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
        n_vox = unique_voxels.shape[0]
        voxel_sums = np.zeros((n_vox, 3), dtype=np.float64)
        np.add.at(voxel_sums, inverse, xyz)
        voxel_counts = np.bincount(inverse).astype(np.float64)
        voxel_centroids = voxel_sums / voxel_counts[:, None]
        cluster_input = voxel_centroids
        index_map = inverse
    else:
        cluster_input = xyz
        index_map = None

# Podle průměrné vzdálenosti - blízko husté body, daleko řídké.
    min_cluster_size = int(np.clip(10 + avg_dist / 4, 10, 40))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        metric="euclidean",
        cluster_selection_epsilon=eps,
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    ).fit(cluster_input)

    labels = clusterer.labels_
    probs = clusterer.probabilities_

    # Rozšíření voxelových výsledků zpět na původní body.
    if index_map is not None:
        point_labels = labels[index_map]
        point_probs = probs[index_map]
    else:
        point_labels = labels
        point_probs = probs

    valid_mask = (point_labels != -1) & (point_probs >= 0.15) # Odstranění šumu.

    clusters: List[np.ndarray] = []
    unique_labels = np.unique(point_labels[valid_mask])
    for lbl in unique_labels:
        if lbl == -1:
            continue
        idx = np.where(valid_mask & (point_labels == lbl))[0]
        if len(idx) > 0:
            clusters.append(idx)

    return clusters
# Vrací seznam clusterů, každý cluster = pole indexů bodů.

# Filtr rozměrů shluku - odstraní nereálné objekty.
def filter_cluster(pts: np.ndarray) -> bool:
    h = pts[:, 2].max() - pts[:, 2].min()  # výška
    w = pts[:, 1].max() - pts[:, 1].min()  # šířka
    d = pts[:, 0].max() - pts[:, 0].min()  # hloubka

    if not ((0.5 <= h <= 3.5) and (0.4 <= w <= 3.0) and (0.5 <= d <= 5.0)):
        return False

    if d < 0.3:
        return False

    volume = max(h * w * d, 0.001) # Hustota bodů - reálné objekty mají hustá mračna.
    density = len(pts) / volume
    if density < 5.0:
        return False

    z_min = pts[:, 2].min() # Nejnižší bod vysoko nad vozovkou -> převis.
    if z_min > 0.5:
        return False

    return True

# Non-Maximum Suppression(NMS) - odstraní duplicitní rámečky.
# Seřadí detekce podle confidence, ponechá nejlepší a odstraní všechny které se s ní překrývají (IoU > prahu).
def apply_nms(detections: List[Dict], iou_thr: float = 0.5) -> List[Dict]:
    if len(detections) <= 1:
        return detections

    def iou(b1, b2):
        xi = max(b1[0], b2[0]); yi = max(b1[1], b2[1])
        xa = min(b1[2], b2[2]); ya = min(b1[3], b2[3])
        if xa <= xi or ya <= yi: return 0.0
        inter = (xa-xi)*(ya-yi)
        union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
        return inter/union if union > 0 else 0.0

# Deláme sorting v sestupním pořadí.
    sorted_d = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    kept = []
    while sorted_d:
        best = sorted_d.pop(0)
        kept.append(best)
        sorted_d = [d for d in sorted_d if iou(best['bbox'], d['bbox']) < iou_thr]
    return kept


# Pipeline LiDAR: odstraní vozovku -> aplikuje masku kamery -> shlukuje pomocí HDBSCAN -> filtruje podle rozměrů.
# Vrací páry (3D body shluku, 2D pixely shluku).
def get_lidar_clusters(lidar_points, points_2d, mask):
    ng = remove_ground(lidar_points)
    if len(mask) == len(lidar_points):
        ng &= mask
    ng_pts = lidar_points[ng]
    ng_2d  = points_2d[ng]
    return [
        (ng_pts[idx], ng_2d[idx])
        for idx in cluster_lidar(ng_pts)
        if len(idx) >= 5 and filter_cluster(ng_pts[idx])
    ]


# --------------------------------------------------------------------------------------------------------------------
# Decision-Level Fusion
# Kamera a LiDAR detekují nezávisle, výsledky se párují pomocí skóre: 0.6 × IoU + 0.4 × blízkost středů rámečků.
# Výsledek má tři typy: fused / image_only / lidar_only.
# --------------------------------------------------------------------------------------------------------------------
class DecisionLevelFusion:
    def __init__(self, yolo_model: YOLO):
        self.yolo   = yolo_model
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # YOLOv8 na celém snímku, přísný práh confidence 0.55 (minimalizace FP).
    def detect_objects_image(self, image) -> List[Dict]:
        detections = []
        for result in self.yolo(image, device=self.device, verbose=False):
            if result.boxes is None: continue
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                cls = int(cls); conf = float(conf) #cls = třída COCO
                if cls not in TARGET_CLASSES or conf < 0.55: continue
                detections.append({
                    'bbox': list(map(int, box.cpu().numpy())),
                    'confidence': conf, 'source': 'image', 'class': COCO_TO_KITTI[cls],
                })
        return detections


    # Detekce čistě z LiDARu - confidence podle počtu bodů, penalta pro vzdálené objekty. Třída 'Object' - LiDAR neumí rozlišit Car/Pedestrian/Cyclist.
    # Na konci aplikuje NMS pro odstranění duplicit.
    def detect_objects_lidar(self, lidar_points, points_2d, mask) -> List[Dict]:
        detections = []
        for cp, c2 in get_lidar_clusters(lidar_points, points_2d, mask): #cp - 3d body clustera, c2 - 2d projekce bodu z lidara na obrazu
            center   = np.mean(cp[:, :3], axis=0)
            distance = float(np.sqrt(np.sum(center**2)))
            conf     = min(1.0, 0.3 + len(cp)/100.0) * (0.8 if distance > 50 else 1.0)
            # Kolik je celkem bodu v clusteru, 0,3+10bodu/100 = 0,4 conf. penalty pro větši vdal >50.
            detections.append({
                'bbox': [
                    max(0, int(c2[:,0].min())-15),
                    max(0, int(c2[:,1].min())-15), #2D projekce + okraj 15px.
                    min(1242, int(c2[:,0].max())+15),
                    min(375, int(c2[:,1].max())+15)],
                'confidence': conf, 'source': 'lidar',
                'class': 'Object', 'distance_m': distance,
            })
        return apply_nms(detections)

    # Greedy párování: kamerové detekce řadíme podle confidence, hledáme nejlepší LiDAR protějšek.
    def fuse_decisions(self, image_dets: List[Dict], lidar_dets: List[Dict]) -> List[Dict]:
        fused = []
        matched_lidar = set()

        sorted_image = sorted(
            enumerate(image_dets),
            key=lambda x: x[1].get('confidence', 0),
            reverse=True
        )
        for orig_idx, img in sorted_image:
            best_j, best_score = -1, 0.0
            for j, lid in enumerate(lidar_dets):
                if j in matched_lidar: continue
                s = self._match_score(img['bbox'], lid['bbox'])
                if s > best_score: best_score, best_j = s, j

            if best_score > 0.20 and best_j >= 0: # Spárováno -> vážená confidence: 70 % kamera (klasifikace) + 30 % LiDAR (geometrie).
                lid = lidar_dets[best_j]
                det = {**img, 'source': 'fused',
                       'confidence': 0.7*img['confidence'] + 0.3*lid['confidence']}
                if 'distance_m' in lid: det['distance_m'] = lid['distance_m']
                fused.append(det); matched_lidar.add(best_j)
            else: # Bez LiDAR protějšku -> penalta confidence.
                fused.append({**img, 'source': 'image_only',
                               'confidence': img['confidence']*0.7})

        for j, lid in enumerate(lidar_dets): # Nespárované LiDAR detekce -> lidar_only.
            if j not in matched_lidar:
                fused.append({**lid, 'source': 'lidar_only',
                               'confidence': lid['confidence']*0.7})
        return fused

    # Výpočet IoU (průnik/sjednocení) dvou rámečků.
    @staticmethod
    def _iou(b1, b2) -> float:
        xi = max(b1[0],b2[0]); yi = max(b1[1],b2[1])
        xa = min(b1[2],b2[2]); ya = min(b1[3],b2[3])
        if xa<=xi or ya<=yi: return 0.0
        inter = (xa-xi)*(ya-yi)
        union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
        return inter/union if union>0 else 0.0

    # Zda oba rámečky popisují stejný objekt, 60% z IoU, 40% z shoda dvou středů (centru). Limit 150px.
    def _match_score(self, b1, b2) -> float:
        cx = math.sqrt(((b1[0]+b1[2])/2-(b2[0]+b2[2])/2)**2 + ((b1[1]+b1[3])/2-(b2[1]+b2[3])/2)**2)
        return 0.6*self._iou(b1,b2) + 0.4*max(0.0, 1-cx/150)

# --------------------------------------------------------------------------------------------------------------------
# Feature-Level Fusion
# LiDAR určuje KDE hledat (oblast zájmu - ROI), YOLOv8 klasifikuje pouze uvnitř výřezu snímku - jeden objekt na shluk.
# --------------------------------------------------------------------------------------------------------------------
class FeatureLevelFusion:
    def __init__(self, yolo_model: YOLO):
        self.yolo   = yolo_model
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Pro každý LiDAR shluk: vyřízne oblast z obrazu (ROI + okraj 25px), spustí YOLO, převede souřadnice zpět.
    def detect_objects(self, image, lidar_points, points_2d, mask, calib=None) -> List[Dict]:
        detections = []
        H, W = image.shape[:2]

        for cp, c2 in get_lidar_clusters(lidar_points, points_2d, mask):
            pad = 60 # YOLO potřebuje kontext okolí.
            x1 = max(0, int(c2[:,0].min())-pad); y1 = max(0, int(c2[:,1].min())-pad)
            x2 = min(W, int(c2[:,0].max())+pad); y2 = min(H, int(c2[:,1].max())+pad)

            if (x2-x1) < 15 or (y2-y1) < 15: continue # příliš malý výřez <15 [px].

            crop = image[y1:y2, x1:x2]

            min_side = 128 # Zvětšení malých výřezů - YOLO horší detekuje miniaturní vstupy.
            ch, cw = crop.shape[:2]
            if ch < min_side or cw < min_side:
                scale = max(min_side / ch, min_side / cw)
                crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)))

            for result in self.yolo(crop, device=self.device, verbose=False):
                if result.boxes is None: continue
                for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    cls = int(cls); conf = float(conf)
                    if cls not in TARGET_CLASSES or conf < 0.25: continue # Nižší práh confidence <0.25 kvůli menšímu kontextu výřezu.

                    bx1 = int(box[0])+x1; by1 = int(box[1])+y1
                    bx2 = int(box[2])+x1; by2 = int(box[3])+y1
                    final_bbox = [bx1, by1, bx2, by2]

                    center   = np.mean(cp[:,:3], axis=0)
                    distance = float(np.sqrt(np.sum(center**2)))

                    detections.append({
                        'bbox': final_bbox, 'confidence': conf,
                        'source': 'feature_fused', 'class': COCO_TO_KITTI[cls],
                        'distance_m': distance,
                    })
                    break  # Jeden shluk = jeden objekt.

        return apply_nms(detections)
