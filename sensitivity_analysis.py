import sys, os, math, time
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from data_loader import KITTIDataLoader
import config

N_FRAMES = 1500
RESULTS_FILE = "sensitivity_results.csv"

# IoU (Intersection over Union) — překryv dvou bboxů [0.0, 1.0].
def _iou(b1, b2):
    xi=max(b1[0],b2[0]); yi=max(b1[1],b2[1])
    xa=min(b1[2],b2[2]); ya=min(b1[3],b2[3])
    if xa<=xi or ya<=yi: return 0.0
    inter=(xa-xi)*(ya-yi)
    union=(b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter
    return inter/union if union>0 else 0.0

# Parametrizovaná verze _match_score z DecisionLevelFusion - umožňuje měnit váhu IoU a normalizaci vzdálenosti středů.
def match_score_param(b1, b2, iou_weight, center_norm):
    cx=math.sqrt(((b1[0]+b1[2])/2-(b2[0]+b2[2])/2)**2+
                 ((b1[1]+b1[3])/2-(b2[1]+b2[3])/2)**2)
    return iou_weight*_iou(b1,b2)+(1-iou_weight)*max(0.0,1-cx/center_norm)

# Spustí kompletní Decision-Level Fusion pipeline s konkrétními hodnotami parametrů a vrátí kumulativní P/R/F1 přes všechny zpracované snímky.

def compute_metrics(loader, yolo, n_frames,
                    iou_weight=0.6, smatch_threshold=0.20,
                    center_norm=150.0, cam_weight=0.7, z_threshold=-1.4):

    from fusion_methods import cluster_lidar, filter_cluster, apply_nms
    COCO_TO_KITTI={0:'Pedestrian',1:'Cyclist',2:'Car',3:'Cyclist',7:'Car'}
    TARGET_CLASSES=list(COCO_TO_KITTI.keys())
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    lidar_weight=1.0-cam_weight
    tp_total=fp_total=fn_total=0

    for i in range(min(n_frames, len(loader.image_files))):
        try:
            image, lidar, calib = loader.get_sample(i)
            pts2d, mask = loader.project_lidar_to_image(lidar, calib)
            gt = loader.load_labels(i)
        except Exception:
            continue
        if not gt: continue

        # Detekce kamerou (YOLOv8).
        img_dets=[]
        for result in yolo(image, device=device, verbose=False):
            if result.boxes is None: continue
            for box,conf,cls in zip(result.boxes.xyxy,result.boxes.conf,result.boxes.cls):
                c=int(cls); f=float(conf)
                if c not in TARGET_CLASSES or f<0.55: continue
                img_dets.append({'bbox':list(map(int,box.cpu().numpy())),
                                 'confidence':f,'class':COCO_TO_KITTI[c]})

        # Detekce LiDARem (s parametrizovaným z_threshold).
        ng = lidar[:,2] > z_threshold           # Odstranění vozovky.
        if len(mask)==len(lidar): ng &= mask    # Pouze body v zorném poli kamery.
        ng_pts=lidar[ng]; ng_2d=pts2d[ng]
        lidar_dets=[]
        for idx in cluster_lidar(ng_pts):
            if len(idx)>=5 and filter_cluster(ng_pts[idx]):
                cp=ng_pts[idx]; c2=ng_2d[idx]
                center=np.mean(cp[:,:3],axis=0)
                dist=float(np.sqrt(np.sum(center**2)))
                conf=min(1.0,0.3+len(cp)/100.0)*(0.8 if dist>50 else 1.0)
                lidar_dets.append({
                    'bbox':[max(0,int(c2[:,0].min())-15),
                            max(0,int(c2[:,1].min())-15),
                            min(1242,int(c2[:,0].max())+15),
                            min(375,int(c2[:,1].max())+15)],
                    'confidence':conf,'distance_m':dist})
        lidar_dets=apply_nms(lidar_dets)

        # Greedy párování kamera <-> LiDAR.
        matched_lidar=set(); fused_dets=[]
        sorted_img=sorted(enumerate(img_dets),
                          key=lambda x:x[1]['confidence'],reverse=True)
        for _,img in sorted_img:
            best_j,best_s=-1,0.0
            for j,lid in enumerate(lidar_dets):
                if j in matched_lidar: continue
                s=match_score_param(img['bbox'],lid['bbox'],iou_weight,center_norm)
                if s>best_s: best_s,best_j=s,j
            if best_s>smatch_threshold and best_j>=0:
                fused_dets.append({
                    'bbox':img['bbox'],'class':img['class'],
                    'confidence':cam_weight*img['confidence']+
                                 lidar_weight*lidar_dets[best_j]['confidence']})
                matched_lidar.add(best_j)

        if not fused_dets: fn_total+=len(gt); continue

        # Vyhodnocení vs. GT (IoU ≥ 0.5).
        matched_gt=set(); matched_det=set()
        for di in sorted(range(len(fused_dets)),
                         key=lambda k:fused_dets[k]['confidence'],reverse=True):
            best_iou,best_gi=0.0,-1
            for gi,gt_obj in enumerate(gt):
                if gi in matched_gt: continue
                v=_iou(fused_dets[di]['bbox'],gt_obj['bbox'])
                if v>best_iou: best_iou,best_gi=v,gi
            if best_iou>=0.5 and best_gi>=0:
                matched_gt.add(best_gi); matched_det.add(di)

        tp=len(matched_gt); fp=len(fused_dets)-tp; fn=len(gt)-tp # Akumulace TP/FP/FN přes všechny snímky
        tp_total+=tp; fp_total+=fp; fn_total+=fn

    # Finální P/R/F1, 1e-9 chrání před dělením nulou.
    prec=tp_total/(tp_total+fp_total+1e-9)
    rec=tp_total/(tp_total+fn_total+1e-9)
    f1=2*prec*rec/(prec+rec+1e-9)
    return {'precision':round(prec,3),'recall':round(rec,3),'f1':round(f1,3),
            'tp':tp_total,'fp':fp_total,'fn':fn_total}


# Pro každou skupinu parametrů (A–E) postupně mění jednu hodnotu a měří dopad na F1 oproti baseline. Výsledky -> CSV + souhrnná tabulka v konzoli
def main():
    print("="*65)
    print("CITLIVOSTNI ANALYZA PARAMETRU - Decision-level Fusion")
    print(f"Pocet kadru: {N_FRAMES}")
    print("="*65)

    loader=KITTIDataLoader(config.KITTI_TRAINING_PATH)
    if not loader.image_files or not loader.has_labels:
        print("CHYBA: Data nebo GT anotace nenalezeny."); return

    print("\nNacitam YOLOv8...")
    yolo=YOLO('yolov8n.pt')
    print("YOLOv8 pripraven.\n")

    # Baseline = hodnoty aktuálně používané v fusion_methods.py.
    BASELINE=dict(iou_weight=0.6, smatch_threshold=0.20,
                  center_norm=150.0, cam_weight=0.7, z_threshold=-1.4)
    rows=[]

    # Pomocná funkce - spustí jednu variantu, vypíše výsledek a uloží do CSV řádku.
    def run(label, group, params):
        t0=time.time()
        print(f"  {label:<42}",end='',flush=True)
        m=compute_metrics(loader,yolo,N_FRAMES,**params)
        print(f"F1={m['f1']:.3f}  P={m['precision']:.3f}"
              f"  R={m['recall']:.3f}  [{time.time()-t0:.0f}s]")
        rows.append({'skupina':group,'varianta':label,**params,**m})

    # [A]
    print("[A] Vahy match_score — iou_weight / center_weight")
    print("-"*65)
    for w in [0.5, 0.6, 0.7]:
        run(f"IoU={w:.1f} / center={1-w:.1f}", 'A', {**BASELINE,'iou_weight':w})

    # [B]
    print("\n[B] Prah parovani smatch")
    print("-"*65)
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30]:
        run(f"smatch > {thr:.2f}", 'B', {**BASELINE,'smatch_threshold':thr})

    # [C]
    print("\n[C] Normalizace vzdalenosti stredu (px)")
    print("-"*65)
    for norm in [100, 150, 200]:
        run(f"center_norm = {norm} px", 'C', {**BASELINE,'center_norm':float(norm)})

    # [D]
    print("\n[D] Vahy confidence kamera / LiDAR")
    print("-"*65)
    for cw in [0.5, 0.6, 0.7, 0.8]:
        run(f"kamera={cw:.1f} / LiDAR={1-cw:.1f}", 'D', {**BASELINE,'cam_weight':cw})

    # [E]
    print("\n[E] Prah vysky pro odstraneni vozovky (m)")
    print("-"*65)
    for z in [-1.2, -1.4, -1.6]:
        run(f"z_threshold = {z} m", 'E', {**BASELINE,'z_threshold':z})

    df=pd.DataFrame(rows)
    df.to_csv(RESULTS_FILE,index=False)

    print("\n"+"="*65)
    print("SOUHRN — odchylka F1 od baseline")
    print("="*65)
    bl=df[(df['iou_weight']==0.6)&(df['smatch_threshold']==0.20)&
          (df['center_norm']==150.0)&(df['cam_weight']==0.7)&
          (df['z_threshold']==-1.4)]
    if not bl.empty:
        bl_f1=bl['f1'].values[0]
        print(f"\n  Baseline F1 = {bl_f1:.3f}\n")
        for _,row in df.iterrows():
            d=row['f1']-bl_f1
            tag="  [baseline]" if abs(d)<0.001 else (f"  +{d:.3f}" if d>0 else f"  {d:.3f}")
            print(f"  {row['varianta']:<42} F1={row['f1']:.3f}{tag}")

    print(f"\nVysledky ulozeny: {RESULTS_FILE}")
    print("="*65)

if __name__=="__main__":
    main()