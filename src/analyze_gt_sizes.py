import sys, os
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from data_loader import KITTIDataLoader
import config

N_FRAMES = 1500


# Validační skript - měří skutečné rozměry GT objektů v KITTI a ověřuje, zda meze v filter_cluster() (fusion_methods.py) pokrývají reálnou populaci.
def main():
    print("="*60)
    print("ANALYZA ROZMERU GT OBJEKTU — KITTI")
    print(f"Pocet snimku: {N_FRAMES}")
    print("="*60)

    loader = KITTIDataLoader(config.KITTI_TRAINING_PATH)

    # Sběr rozměrů pro každou sledovanou třídu.
    stats = {
        'Car':        {'h': [], 'w': [], 'd': []},
        'Pedestrian': {'h': [], 'w': [], 'd': []},
        'Cyclist':    {'h': [], 'w': [], 'd': []},
    }

    for i in range(min(N_FRAMES, len(loader.image_files))):
        # Čteme raw label soubor - load_labels() vrací jen 2D bbox, ne 3D rozměry.
        if i >= len(loader.label_files):
            continue
        with open(loader.label_files[i], 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                obj_class  = parts[0]
                truncated  = float(parts[1])
                occluded   = int(parts[2])

                # Stejný filtr jako v load_labels - vyloučí nespolehlivé GT.
                if obj_class not in stats:
                    continue
                if truncated > 0.8 or occluded > 2:
                    continue

                # KITTI label format: parts[8]=h (výška), parts[9]=w (šířka), parts[10]=l (délka).
                h = float(parts[8])
                w = float(parts[9])
                d = float(parts[10])

                stats[obj_class]['h'].append(h)
                stats[obj_class]['w'].append(w)
                stats[obj_class]['d'].append(d)

    # Statistický souhrn pro každou třídu - min/max/mean a percentily 5/95, (p5/p95 jsou robustnější než min/max — odolávají odlehlým hodnotám).
    print()
    for cls, data in stats.items():
        if not data['h']:
            continue
        h = np.array(data['h'])
        w = np.array(data['w'])
        d = np.array(data['d'])

        print(f"[{cls}]  n={len(h)} objektu")
        print(f"  Vyska (h): min={h.min():.2f}  "
              f"max={h.max():.2f}  "
              f"mean={h.mean():.2f}  "
              f"p5={np.percentile(h,5):.2f}  "
              f"p95={np.percentile(h,95):.2f}")
        print(f"  Sirka (w): min={w.min():.2f}  "
              f"max={w.max():.2f}  "
              f"mean={w.mean():.2f}  "
              f"p5={np.percentile(w,5):.2f}  "
              f"p95={np.percentile(w,95):.2f}")
        print(f"  Hloubka(d): min={d.min():.2f}  "
              f"max={d.max():.2f}  "
              f"mean={d.mean():.2f}  "
              f"p5={np.percentile(d,5):.2f}  "
              f"p95={np.percentile(d,95):.2f}")
        print()

    # Připomenutí mezí použitých ve filter_cluster().
    print("="*60)
    print("POROVNANI S FILTREM filter_cluster:")
    print("  h in <0.5, 3.5> m")
    print("  w in <0.4, 3.0> m")
    print("  d in <0.5, 5.0> m")
    print("="*60)
    print()

    # Kolik GT objektů by skutečně prošlo geometrickým filtrem.
    print("POKRYTI FILTREM (% GT objektu ktere splnuji meze):")
    for cls, data in stats.items():
        if not data['h']:
            continue
        h = np.array(data['h'])
        w = np.array(data['w'])
        d = np.array(data['d'])

        mask = (
            (h >= 0.5) & (h <= 3.5) &
            (w >= 0.4) & (w <= 3.0) &
            (d >= 0.5) & (d <= 5.0)
        )
        pct = mask.sum() / len(h) * 100
        print(f"  {cls:<12}: {mask.sum()}/{len(h)} "
              f"= {pct:.1f}% objektu splnuje geometricky filtr")

    print("="*60)

if __name__ == "__main__":
    main()
