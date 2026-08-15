import numpy as np
import cv2
import config
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


# Vykresluje detekce a LiDAR body a sestavuje 6-panelový srovnávací graf.
class Visualizer:

    # Barevná paleta zdrojů detekce (BGR pro OpenCV).
    def __init__(self):
        self.colors = {
            'image': (0, 255, 0),            # zelená  - kamera YOLOv8
            'lidar': (255, 0, 0),            # modrá   - LiDAR shluk
            'fused': (0, 0, 255),            # červená - Decision Fusion (fused)
            'feature_fused': (0, 165, 255),  # oranžová - Feature Fusion
            'image_only': (0, 255, 255),     # žlutá   - pouze kamera
            'lidar_only': (255, 0, 255)      # fialová - pouze LiDAR
        }

    # Promítne viditelné LiDAR body do snímku jako barevné tečky.
    # Barva podle vzdálenosti: modrá (blízko) -> žlutá (střední) -> červená (50 m+).
    def visualize_lidar_on_image(self, image: np.ndarray,
                                 points: np.ndarray,
                                 points_2d: np.ndarray,
                                 mask: np.ndarray) -> np.ndarray:

        vis_image = image.copy()
        visible_points_2d = points_2d[mask]
        visible_points = points[mask]

        if len(visible_points_2d) == 0:
            return vis_image

        # Normalizace vzdáleností na rozsah [0, 1] pro barevnou mapu.
        distances = np.sqrt(np.sum(visible_points[:, :3]**2, axis=1))
        distances_norm = np.clip((distances - 0.0) / (50.0 - 0.0), 0.0, 1.0)

        # plt.cm.jet: hodnota [0,1] -> RGB barva (modrá -> červená)
        colors = plt.cm.jet(distances_norm)[:, :3] * 255
        colors = colors.astype(np.uint8)

        for i, (point_2d, color) in enumerate(zip(visible_points_2d, colors)):
            x, y = int(point_2d[0]), int(point_2d[1])
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(vis_image, (x, y), 2, tuple(map(int, color)), -1)

        return vis_image

    # Vykreslí bboxy detekcí - barva podle zdroje (viz self.colors), popisek = třída + confidence + vzdálenost.
    def visualize_detections(self, image: np.ndarray,
                            detections: List[Dict],
                            thickness: int = 2) -> np.ndarray:
        vis_image = image.copy()

        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            source = det['source']
            obj_class = det.get('class', 'Unknown')
            distance = det.get('distance_m', None)

            x_min, y_min, x_max, y_max = bbox
            color = self.colors.get(source, (255, 255, 255)) # Bílá pro neznámý zdroj.

            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, thickness)

            if distance is not None:
                label = f"{obj_class} {confidence:.2f} | {distance:.1f}m"
            else:
                label = f"{obj_class} {confidence:.2f}"

            font_scale = 0.5
            font_thickness = 1

            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Barevné pozadí pod textem pro lepší čitelnost
            cv2.rectangle(vis_image,
                         (x_min, y_min - text_height - baseline - 5),
                         (x_min + text_width, y_min),
                         color, -1)

            cv2.putText(vis_image, label, (x_min, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        return vis_image

    # Sestaví 6-panelový srovnávací graf pro jeden snímek:
    # [LiDAR projekce] [Kamera YOLOv8] [Feature Fusion];
    # [LiDAR shluky] [Decision Fusion] [Porovnání metod];
    # Uloží se jako PNG do složky results.
    def create_comparison_plot(self, image: np.ndarray,
                              lidar_points: np.ndarray,
                              points_2d: np.ndarray,
                              mask: np.ndarray,
                              image_detections: List[Dict],
                              lidar_detections: List[Dict],
                              fused_detections: List[Dict],
                              feature_detections: List[Dict] = None) -> plt.Figure:

        if feature_detections is None:
            feature_detections = []

        fig, axes = plt.subplots(2, 3, figsize=(30, 14))

        # Panel 1: LiDAR body promítnuté na fotografii (barevně podle vzdálenosti).
        ax1 = axes[0, 0]
        if config.SHOW_LIDAR_PROJECTION:
            vis_lidar = self.visualize_lidar_on_image(image, lidar_points, points_2d, mask)
            ax1.imshow(cv2.cvtColor(vis_lidar, cv2.COLOR_BGR2RGB))
            ax1.set_title('LiDAR projekce', fontsize=16, fontweight='bold')
        else:
            ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax1.set_title('Původní snímek', fontsize=16, fontweight='bold')
        ax1.axis('off')

        # Panel 2: Detekce kamery YOLOv8 (zelené rámečky).
        ax2 = axes[0, 1]
        vis_image_det = self.visualize_detections(image, image_detections, thickness=2)
        ax2.imshow(cv2.cvtColor(vis_image_det, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Kamera YOLOv8 ({len(image_detections)})', fontsize=16, fontweight='bold')
        ax2.axis('off')

        # Panel 3: Feature Fusion (oranžové rámečky - LiDAR určil KDE, YOLO určila CO).
        ax3 = axes[0, 2]
        vis_feature = self.visualize_detections(image, feature_detections, thickness=3)
        ax3.imshow(cv2.cvtColor(vis_feature, cv2.COLOR_BGR2RGB))
        ax3.set_title(f'Feature Fusion ({len(feature_detections)})', fontsize=16, fontweight='bold')
        ax3.axis('off')

        # Panel 4: LiDAR shluky jako rámečky (modré bboxy).
        ax4 = axes[1, 0]
        vis_lidar_det = self.visualize_detections(image, lidar_detections, thickness=2)
        ax4.imshow(cv2.cvtColor(vis_lidar_det, cv2.COLOR_BGR2RGB))
        ax4.set_title(f'LiDAR klastery ({len(lidar_detections)})', fontsize=16, fontweight='bold')
        ax4.axis('off')

        # Panel 5: Decision Fusion - v titulku počet skutečně sloučených (source == 'fused').
        ax5 = axes[1, 1]
        fused_only = [d for d in fused_detections if d.get('source') == 'fused']
        vis_fused = self.visualize_detections(image, fused_detections, thickness=2)
        ax5.imshow(cv2.cvtColor(vis_fused, cv2.COLOR_BGR2RGB))
        ax5.set_title(f'Decision Fusion ({len(fused_only)})', fontsize=16, fontweight='bold')
        ax5.axis('off')

        # Panel 6: Přímé srovnání Feature (oranžová) × Decision (červená).
        ax6 = axes[1, 2]
        vis_comparison = image.copy()

        for det in feature_detections:
            bbox = det['bbox']
            confidence = det['confidence']
            obj_class = det.get('class', 'Unknown')
            distance = det.get('distance_m', None)
            x_min, y_min, x_max, y_max = bbox
            color = (0, 165, 255)
            cv2.rectangle(vis_comparison, (x_min, y_min), (x_max, y_max), color, 3)
            label = f"{obj_class} {confidence:.2f}"
            if distance is not None:
                label += f" | {distance:.1f}m"
            cv2.putText(vis_comparison, label, (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for det in fused_detections:
            if det.get('source') == 'fused':
                bbox = det['bbox']
                confidence = det['confidence']
                obj_class = det.get('class', 'Unknown')
                distance = det.get('distance_m', None)
                x_min, y_min, x_max, y_max = bbox
                color = (0, 0, 255)
                cv2.rectangle(vis_comparison, (x_min, y_min), (x_max, y_max), color, 2)
                label = f"{obj_class} {confidence:.2f}"
                if distance is not None:
                    label += f" | {distance:.1f}m"
                cv2.putText(vis_comparison, label, (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Legenda v levém horním rohu panelu 6.
        legend_y = 30
        cv2.rectangle(vis_comparison, (10, legend_y - 20), (300, legend_y + 50), (0, 0, 0), -1)
        cv2.putText(vis_comparison, "Feature Fusion", (20, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(vis_comparison, "Decision Fusion", (20, legend_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        ax6.imshow(cv2.cvtColor(vis_comparison, cv2.COLOR_BGR2RGB))
        ax6.set_title('Porovnání metod', fontsize=16, fontweight='bold')
        ax6.axis('off')

        plt.tight_layout()
        return fig