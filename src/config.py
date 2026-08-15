# Kořenová složka datasetu KITTI (musí obsahovat podadresáře data_object_*)
KITTI_DATA_PATH      = "../Sample_data/DATASET"
KITTI_TRAINING_PATH  = "../Sample_data/DATASET"

# Relativní cesty k jednotlivým částem datasetu (obraz, LiDAR, kalibrace, GT)
IMAGE_PATH = "data_object_image_2/training/image_2"
LIDAR_PATH = "data_object_velodyne/training/velodyne"
CALIB_PATH = "data_object_calib/training/calib"
LABEL_PATH = "data_object_label_2/training/label_2"

# Rozlišení snímků KITTI
IMAGE_WIDTH  = 1242
IMAGE_HEIGHT = 375

# Oba parametry musí být True pro uložení výstupů na disk
VISUALIZE    = True # TRUE/FALSE - vytvoří 6-panelový srovnávací graf pro každý snímek ve složce "results"
SAVE_RESULTS = True # TRUE/FALSE - uloží výsledné grafy jako PNG do složky
RESULTS_PATH = "results"

# Přepínače jednotlivých detekčních metod
SHOW_LIDAR_PROJECTION = True   # LiDAR projekce to Kamera
USE_CAMERA_DETECTION  = True   # YOLOv8
USE_LIDAR_DETECTION   = True   # HDBSCAN Clustering
USE_DECISION_FUSION   = True   # Decision-level
USE_FEATURE_FUSION    = True   # Feature-level

# Vyhodnocení přesnosti oproti ground truth anotacím KITTI
EVALUATE_WITH_GT  = True
GT_IOU_THRESHOLD  = 0.5 #sjednocení dvou ohraničujících boxů
GT_CLASSES        = ['Car', 'Pedestrian', 'Cyclist']

# Vzdálenostní pásma [m]
DISTANCE_BINS = [0, 20, 40, 80]

