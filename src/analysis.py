from typing import List, Dict
import config


# IoU (Intersection over Union) - překryv dvou bboxů, hodnota [0.0, 1.0].
def _iou(b1, b2) -> float:
    xi = max(b1[0], b2[0]); yi = max(b1[1], b2[1]) # Levý-horní roh průniku.
    xa = min(b1[2], b2[2]); ya = min(b1[3], b2[3]) # Pravý-dolní roh průniku.
    if xa <= xi or ya <= yi:
        return 0.0 # bboxy se neprotínají.
    inter = (xa - xi) * (ya - yi)
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / union if union > 0 else 0.0

# Prázdná struktura metrik - vrací se v případě chyby nebo neznámé metody.
def _empty() -> Dict:
    return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0, 'n_gt': 0}

# Vyhodnocuje přesnost čtyř metod (camera, lidar, decision_fusion, feature_fusion) vs. KITTI ground truth. Sbírá TP/FP/FN podle třídy a vzdálenostního pásma.
class FusionAnalyzer:
    METHODS = ['camera', 'lidar', 'decision_fusion', 'feature_fusion']

    def __init__(self):
        self.iou_thr   = config.GT_IOU_THRESHOLD # min. IoU pro spárování (0.5).
        self.dist_bins = config.DISTANCE_BINS    # [0, 20, 40, 80] [m].
        self.classes   = config.GT_CLASSES       # Car / Pedestrian / Cyclist.
        self.accum     = {m: self._init_accum() for m in self.METHODS} #[metoda][třída][pásmo] -> {tp, fp, fn, n_gt}

    # Sestaví štítky vzdálenostních pásem: "0-20m", "20-40m", "40-80m", ">80m".
    def _bin_labels(self) -> List[str]:
        bins = self.dist_bins
        labels = [f"{bins[i]}-{bins[i+1]}m" for i in range(len(bins)-1)]
        labels.append(f">{bins[-1]}m")
        return labels

    # Přiřadí konkrétní vzdálenost odpovídajícímu pásmu.
    def _dist_bin(self, dist: float) -> str:
        bins = self.dist_bins
        for i in range(len(bins)-1):
            if dist < bins[i+1]:
                return f"{bins[i]}-{bins[i+1]}m"
        return f">{bins[-1]}m"

    # Vytvoří prázdné počítadlo pro všechny kombinace (třída × pásmo) + 'ALL' - agregát přes všechny třídy / všechna pásma.
    def _init_accum(self):
        all_bins = self._bin_labels() + ['ALL']
        all_cls  = self.classes + ['ALL']
        return {c: {b: {'tp':0,'fp':0,'fn':0,'n_gt':0} for b in all_bins} for c in all_cls}

    # Vyhodnocení jednoho snímku - spáruje detekce s GT a aktualizuje akumulátor.
    def evaluate_frame(self, method: str,
                       detections: List[Dict],
                       ground_truth: List[Dict]) -> Dict:
        if method not in self.accum:
            return _empty()

        # Greedy párování: detekce od nejvyšší confidence hledá nejlepší volný GT.
        matched_gt  = set()
        matched_det = set()
        det_order   = sorted(range(len(detections)),
                             key=lambda i: detections[i].get('confidence', 0), reverse=True)

        for di in det_order:
            best_iou, best_gi = 0.0, -1
            for gi, gt in enumerate(ground_truth):
                if gi in matched_gt:
                    continue
                v = _iou(detections[di]['bbox'], gt['bbox'])
                if v > best_iou:
                    best_iou, best_gi = v, gi
            if best_iou >= self.iou_thr and best_gi >= 0: # Spárování platí, pouze pokud IoU překročí práh (typicky 0.5).
                matched_gt.add(best_gi); matched_det.add(di)

        tp = len(matched_gt)            # TP = spárované GT.
        fp = len(detections) - tp       # FP = nespárované detekce.
        fn = len(ground_truth) - tp     # FN = nespárované GT.

        # Aktualizace akumulátoru pro každý GT objekt (do TP nebo FN).
        for gi, gt in enumerate(ground_truth):
            gt_cls = gt.get('class', 'ALL')
            gt_bin = self._dist_bin(gt.get('distance_m', 0.0))
            is_tp  = (gi in matched_gt)
            for ck in ([gt_cls, 'ALL'] if gt_cls in self.classes else ['ALL']): # Zapisujeme jak do konkrétní třídy/pásma, tak do agregátu 'ALL'.
                for bk in [gt_bin, 'ALL']:
                    c = self.accum[method][ck][bk]
                    c['n_gt'] += 1
                    if is_tp: c['tp'] += 1
                    else:     c['fn'] += 1

        # Nespárované detekce -> FP.
        for di in range(len(detections)):
            if di not in matched_det:
                det_cls = detections[di].get('class', 'ALL')
                det_bin = self._dist_bin(detections[di].get('distance_m', 0.0))
                for ck in ([det_cls, 'ALL'] if det_cls in self.classes else ['ALL']):
                    for bk in [det_bin, 'ALL']:
                        self.accum[method][ck][bk]['fp'] += 1

        prec = tp / (tp + fp + 1e-9)    # 1e-9 chrání před dělením nulou při prázdných výsledcích.
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2*prec*rec / (prec + rec + 1e-9)
        return {'precision': prec, 'recall': rec, 'f1': f1,
                'tp': tp, 'fp': fp, 'fn': fn, 'n_gt': len(ground_truth)}

    # Kumulativní metriky z akumulátoru pro zvolenou kombinaci (metoda, třída, pásmo).
    def get_metrics(self, method: str, cls: str = 'ALL', dist_bin: str = 'ALL') -> Dict:
        if method not in self.accum:
            return _empty()
        c = self.accum[method].get(cls, {}).get(dist_bin)
        if c is None:
            return _empty()
        tp, fp, fn = c['tp'], c['fp'], c['fn']
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2*prec*rec / (prec + rec + 1e-9)
        return {'precision': prec, 'recall': rec, 'f1': f1,
                'tp': tp, 'fp': fp, 'fn': fn, 'n_gt': c['n_gt']}

    # Tiskne tři tabulky do konzole: celková přesnost, podle třídy, podle vzdálenosti (F1).
    def print_summary(self):
        print("\n" + "="*75)
        print("VÝSLEDKY HODNOCENÍ PŘESNOSTI (vs. KITTI ground truth)")
        print("="*75)
        hdr = f"{'Metoda':<22} {'Prec':>6} {'Recall':>7} {'F1':>6}  {'TP':>5} {'FP':>5} {'FN':>5}"

        def row(method):
            m = self.get_metrics(method, 'ALL', 'ALL')
            return (f"{method:<22} {m['precision']:6.3f} {m['recall']:7.3f} {m['f1']:6.3f}  "
                    f"{m['tp']:5d} {m['fp']:5d} {m['fn']:5d}")

        # Tabulka 1: agregát přes všechny třídy a vzdálenosti.
        print("\n[CELKOVÁ PŘESNOST]")
        print(hdr); print("-"*60)
        for meth in self.METHODS: print(row(meth))

        # Tabulka 2: rozdělení podle tříd (Car / Pedestrian / Cyclist).
        print("\n[PODLE TŘÍDY]")
        for cls in self.classes:
            print(f"\n  {cls}:"); print("  "+hdr); print("  "+"-"*60)
            for meth in self.METHODS:
                m = self.get_metrics(meth, cls, 'ALL')
                print(f"  {meth:<22} {m['precision']:6.3f} {m['recall']:7.3f} {m['f1']:6.3f}  "
                      f"{m['tp']:5d} {m['fp']:5d} {m['fn']:5d}")

        # Tabulka 3: F1 podle vzdálenostních pásem — ukazuje degradaci se vzdáleností.
        print("\n[PODLE VZDÁLENOSTI — F1]")
        bins = self._bin_labels()
        print(f"{'Metoda':<22}" + "".join(f"{b:>13}" for b in bins))
        print("-"*(22+13*len(bins)))
        for meth in self.METHODS:
            row_s = f"{meth:<22}" + "".join(f"{self.get_metrics(meth,'ALL',b)['f1']:13.3f}" for b in bins)
            print(row_s)
        print("="*75)

