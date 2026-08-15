import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import torch

from data_loader import KITTIDataLoader
from fusion_methods import FeatureLevelFusion, DecisionLevelFusion
from visualization import Visualizer
from analysis import FusionAnalyzer
import config


# Pipeline: načtení dat -> detekce -> fúze -> vyhodnocení GT -> vizualizace -> uložení.
def main():
    print("=" * 70)
    print("SENZOROVÁ FÚZE LiDAR A KAMERY")
    print("=" * 70)

    # Kontrola zda složka s daty KITTI existuje.
    if not os.path.exists(config.KITTI_TRAINING_PATH):
        print(f"\nCHYBA: Data KITTI nebyla nalezena v {config.KITTI_TRAINING_PATH}")
        return

    # Načtení datové sady - sestaví seznamy souborů a zkontroluje GT.
    loader = KITTIDataLoader(config.KITTI_TRAINING_PATH)
    if not loader.image_files:
        print("CHYBA: Nebyly nalezeny žádné obrázky!")
        return

    # Jeden model YOLOv8 sdílený oběma metodami - úspora RAM a času načtení.
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\nNačtení YOLOv8 ({'GPU' if 'cuda' in device else 'CPU'})...")
    yolo = YOLO('yolov8n.pt')
    print("YOLOv8 načteno\n")

    # Inicializace metod podle přepínačů v config.py.
    # Decision-level Fusion obstarává také samostatnou detekci kamery a LiDARu.
    decision_fusion = DecisionLevelFusion(yolo) if (config.USE_DECISION_FUSION or
                                                     config.USE_CAMERA_DETECTION or
                                                     config.USE_LIDAR_DETECTION) else None
    feature_fusion  = FeatureLevelFusion(yolo)  if config.USE_FEATURE_FUSION else None
    visualizer      = Visualizer()
    analyzer        = FusionAnalyzer() if config.EVALUATE_WITH_GT and loader.has_labels else None # Analyzátor jen pokud jsou dostupné GT anotace.
    if analyzer:
        print("Ground truth anotace nalezeny - bude provedena kontrola přesnosti\n")
    else:
        print("Ground truth anotace nenalezeny - kontrola přesnosti nebude provedena.")

    # Příprava výstupní složky - smaže staré PNG z předchozího běhu.
    results_path = Path(config.RESULTS_PATH)
    results_path.mkdir(exist_ok=True)
    for f in results_path.glob("result_frame_*.png"):
        f.unlink()

    # Uživatelský vstup: kolik snímků chce zpracovat (1 až max dostupných).
    print(f"Dostupných snímků: {len(loader.image_files)}")
    while True:
        try:
            n = int(input("Kolik snímků zpracovat? "))
            if 1 <= n <= len(loader.image_files): break
            print(f"Zadejte číslo od 1 do {len(loader.image_files)}")
        except ValueError:
            print("Je třeba zadat celé číslo")

    print(f"\nZpracování {n} snímků...\n")

    # Slovník statistik — každý klíč = sloupec v statistics.csv.
    stats = {k: [] for k in [
        'frame', 'lidar_points', 'visible_points',
        'image_detections', 'lidar_detections',
        'fused_detections', 'feature_detections',
        'fused_count', 'processing_time'
    ]}
    ok = fail = 0

    # Zpracování každého snímku
    for i in range(n):
        # Animovaný progress bar v konzoli.
        pct = (i+1)/n*100
        print(f"\r[{'|'*int(pct/2):<50}] {pct:.1f}% ({i+1}/{n})", end='')

        try:
            t0 = time.time()

            # Načtení snímku, LiDAR mračna a kalibračních matic.
            image, lidar, calib = loader.get_sample(i)
            # Projekce LiDAR bodů na obraz kamery -> pixelové souřadnice + maska viditelnosti.
            pts2d, mask = loader.project_lidar_to_image(lidar, calib)

            # Detekce podle aktivních přepínačů v config.py.
            img_dets     = decision_fusion.detect_objects_image(image)                           if config.USE_CAMERA_DETECTION else []
            lidar_dets   = decision_fusion.detect_objects_lidar(lidar, pts2d, mask)              if config.USE_LIDAR_DETECTION  else []
            fused_dets   = decision_fusion.fuse_decisions(img_dets, lidar_dets)                  if config.USE_DECISION_FUSION  else []
            feature_dets = feature_fusion.detect_objects(image, lidar, pts2d, mask, calib)       if (feature_fusion and config.USE_FEATURE_FUSION) else []

            # Počet skutečně sloučených detekcí (source == 'fused', bez image_only/lidar_only).
            fused_count = sum(1 for d in fused_dets if d.get('source') == 'fused')
            elapsed = time.time() - t0

            # Vyhodnocení přesnosti oproti GT - pokud jsou anotace k dispozici.
            if analyzer:
                gt = loader.load_labels(i)
                if gt:
                    if config.USE_CAMERA_DETECTION:
                        analyzer.evaluate_frame('camera', img_dets, gt)
                    if config.USE_LIDAR_DETECTION:
                        analyzer.evaluate_frame('lidar', lidar_dets, gt)
                    if config.USE_DECISION_FUSION:
                        fused_only = [d for d in fused_dets if d.get('source') == 'fused'] # Hodnotíme jen skutečně sloučené detekce.
                        analyzer.evaluate_frame('decision_fusion', fused_only, gt)
                    if config.USE_FEATURE_FUSION:
                        analyzer.evaluate_frame('feature_fusion', feature_dets, gt)

            # Uložení statistik snímku.
            for key, val in zip(stats.keys(), [
                i, len(lidar), int(mask.sum()),
                len(img_dets), len(lidar_dets),
                len(fused_dets), len(feature_dets),
                fused_count, elapsed
            ]):
                stats[key].append(val)

            # Vytvoření a uložení 6-panelového srovnávacího grafu.
            if config.VISUALIZE:
                fig = visualizer.create_comparison_plot(
                    image, lidar, pts2d, mask,
                    img_dets, lidar_dets, fused_dets, feature_dets
                )
                if config.SAVE_RESULTS:
                    fig.savefig(results_path / f"result_frame_{i:06d}.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
            ok += 1


        except Exception as e:
            fail += 1
            print(f"\nChyba snímek {i}: {e}")

    print("\n")

    # Uložení statistik do CSV souboru (jeden řádek = jeden snímek)
    df = pd.DataFrame(stats)
    df.to_csv(results_path / "statistics.csv", index=False)
    print(f"✓ Statistika: {results_path / 'statistics.csv'}")

    # Textová zpráva se shrnutím výsledků
    _write_report(df, results_path, ok, fail, n, analyzer)

    # Tabulka přesnosti v konzoli + uložení do metrics.csv
    if analyzer:
        analyzer.print_summary()
        _save_metrics_csv(analyzer, results_path)

    print(f"\n{'='*70}")
    print(f"Úspěšně: {ok}/{n}   Chyby: {fail}/{n}")
    print(f"Výsledky: {results_path}")
    print('='*70)


# Zapíše textovou zprávu report.txt se shrnutím celého běhu: obecné info, aktivní metody, statistiky LiDAR, průměry detekcí, výkon (FPS) a přesnost vs. GT (pokud je dostupná).
def _write_report(df, path, ok, fail, total, analyzer=None):
    with open(path / "report.txt", 'w', encoding='utf-8') as f:
        def w(s=""): f.write(s + "\n")
        w("="*70); w("ZPRÁVA O ANALÝZE SENZORICKÉ FÚZE"); w("="*70)

        w("\n1. OBECNÉ"); w("-"*70)
        w(f"Celkem: {total}  Úspěšně: {ok}  Chyby: {fail}")
        w(f"Úspěšnost: {ok/total*100:.1f}%")

        w("\n2. AKTIVNÍ METODY"); w("-"*70)
        for flag, name in [
            (config.USE_CAMERA_DETECTION, "Kamera YOLOv8"),
            (config.USE_LIDAR_DETECTION,  "LiDAR HDBSCAN"),
            (config.USE_DECISION_FUSION,  "Decision-level Fusion"),
            (config.USE_FEATURE_FUSION,   "Feature-level Fusion"),
        ]:
            w(f"  {'✓' if flag else '✗'}  {name}")

        w("\n3. LiDAR BODY"); w("-"*70)
        w(f"Průměr bodů LiDAR:    {df.lidar_points.mean():.0f}")
        w(f"Průměr viditelných:   {df.visible_points.mean():.0f}")
        w(f"Podíl viditelných:    {df.visible_points.mean()/df.lidar_points.mean()*100:.1f}%")

        w("\n4. DETEKCE (průměr / snímek)"); w("-"*70)
        w(f"Kamera YOLOv8:           {df.image_detections.mean():.1f}")
        w(f"LiDAR HDBSCAN+NMS:       {df.lidar_detections.mean():.1f}")
        w(f"Decision Fusion celkem:  {df.fused_detections.mean():.1f}")
        w(f"  z toho skutečně fused: {df.fused_count.mean():.1f}")
        w(f"Feature Fusion:          {df.feature_detections.mean():.1f}")

        fps = 1 / df.processing_time.mean()
        w("\n5. VÝKON"); w("-"*70)
        w(f"Průměrný čas / snímek: {df.processing_time.mean():.3f} s")
        w(f"FPS: {fps:.2f}")

        if analyzer:
            w("\n6. PŘESNOST vs. GROUND TRUTH"); w("-"*70)
            for meth in analyzer.METHODS:
                m = analyzer.get_metrics(meth, 'ALL', 'ALL')
                w(f"  {meth:<22}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
                  f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})")
            w("\n  Podle třídy:")
            for cls in analyzer.classes:
                w(f"    {cls}:")
                for meth in analyzer.METHODS:
                    m = analyzer.get_metrics(meth, cls, 'ALL')
                    w(f"      {meth:<22}: F1={m['f1']:.3f}  (TP={m['tp']} FP={m['fp']} FN={m['fn']})")
            w("\n  Podle vzdálenosti (F1):")
            bins = analyzer._bin_labels()
            for meth in analyzer.METHODS:
                parts = [f"{analyzer.get_metrics(meth,'ALL',b)['f1']:.3f}" for b in bins]
                w(f"    {meth:<22}: " + "  ".join(f"{b}={v}" for b,v in zip(bins,parts)))

        w("\n7. ZÁVĚRY"); w("-"*70)
        w("• Feature Fusion: LiDAR určuje oblast, YOLOv8 klasifikuje objekt")
        w("• Decision Fusion: nezávislé detekce sloučeny přes IoU + NMS")
        w(f"• Systém pracuje při ~{fps:.1f} FPS")
        w("="*70)
    print(f"Reporty: {path / 'report.txt'}")


# Uloží kompletní metriky (P/R/F1) do metrics.csv. - řádek = kombinace metoda × třída × vzdálenostní pásmo.
def _save_metrics_csv(analyzer, path):
    rows = []
    bins = analyzer._bin_labels() + ['ALL']
    for meth in analyzer.METHODS:
        for cls in analyzer.classes + ['ALL']:
            for bn in bins:
                m = analyzer.get_metrics(meth, cls, bn)
                rows.append({
                    'method': meth, 'class': cls, 'distance_bin': bn,
                    **m
                })
    pd.DataFrame(rows).to_csv(path / "metrics.csv", index=False)
    print(f"Metriky: {path / 'metrics.csv'}")


if __name__ == "__main__":
    main()
