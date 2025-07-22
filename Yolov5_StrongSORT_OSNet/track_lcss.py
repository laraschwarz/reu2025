import argparse
import collections
from shapely.geometry import Point, Polygon
import os
from tslearn.metrics import lcss
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
import math
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from fastdtw import fastdtw
from scipy.spatial.distance import cosine

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True) if save_vid or save_txt else None # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
        strongsort_list[i].model.warmup()
    outputs = [None] * nr_sources

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

    # Added for tracking first and last coordinates of each track
    coords = {}      # id → list of coordinates
    previous_ids = set()        # ids seen in the prior frame
    crossed_ids = set()        # ids that crossed the intersection
    names = model.names
    id_to_class = {}      # will map each track ID to its class label
    ids_per_frame = []  # will map each frame to the set of IDs seen in that frame
    recent_ids = []  # will store the IDs seen in the last 10 frames

    # count how many objects crossed from west to east
    count = 0
    we = [] # west to east
    ew = [] # east to west
    ne = [] # north to east
    ns = [] # north to south
    sn = [] # south to north

    
    cross_display = [] # messages to display on video

    # base path for objects that cross from west to east, based on car ID:10
    we_path = [(39.5, 229.0), (45.5, 232.0), (52.0, 236.5), (59.5, 242.5), (66.5, 250.0), (73.5, 256.0), (83.5, 264.5), (98.0, 271.5), (111.5, 280.5), (126.5, 287.5), (140.0, 293.5), (155.0, 301.5), (170.0, 308.0), (183.5, 316.0), (200.0, 324.5), (213.5, 332.5), (230.5, 340.0), (246.0, 349.5), (262.0, 357.0), (277.0, 364.5), (293.0, 372.5), (307.0, 381.0), (325.0, 390.0), (342.0, 399.0), (358.0, 407.5), (375.5, 416.0), (392.0, 425.0), (408.5, 432.0), (426.5, 441.5), (441.5, 450.5), (460.5, 459.0), (478.0, 467.5), (496.0, 476.5), (513.5, 486.5), (532.0, 495.0), (550.0, 505.0), (568.0, 515.0), (587.5, 523.5), (606.0, 535.0), (625.5, 544.5), (644.5, 555.0), (664.5, 565.5), (684.0, 575.0), (704.0, 586.0), (724.5, 596.0), (745.5, 608.5), (766.0, 619.0), (787.0, 629.5), (808.5, 641.0), (831.5, 646.5), (853.0, 651.0), (874.5, 656.0), (897.5, 663.0), (918.5, 669.0), (923.5, 675.0), (948.0, 680.5), (962.0, 686.5), (970.5, 692.5), (986.5, 698.0), (1002.0, 705.0), (1019.5, 711.5)]
    we_path_vectors = np.diff(np.array(we_path), axis=0) # compute differences between consecutive points (accounts for parallel paths)
    ns_path = [(890.0, 8.5), (889.5, 9.0), (889.0, 9.0), (888.5, 9.0), (888.5, 9.5), (887.0, 10.0), (886.0, 10.5), (886.0, 11.0), (886.0, 11.5), (885.5, 11.5), (885.0, 11.5), (885.0, 12.0), (884.5, 12.5), (884.0, 12.5), (883.5, 13.0), (883.0, 13.0), (883.0, 13.5), (882.0, 13.5), (881.5, 13.5), (881.0, 14.0), (879.5, 14.5), (879.0, 15.0), (878.5, 15.0), (878.0, 15.0), (877.0, 15.5), (877.5, 16.0), (877.0, 16.0), (876.5, 16.5), (875.5, 17.0), (876.0, 17.5), (875.5, 18.0), (874.5, 18.0), (874.0, 18.5), (874.0, 19.0), (874.0, 19.5), (873.5, 20.0), (872.0, 21.0), (871.5, 21.5), (870.5, 22.0), (870.5, 22.5), (869.5, 23.5), (869.5, 24.0), (867.0, 24.5), (865.5, 25.5), (866.0, 26.5), (866.0, 27.5), (867.0, 28.5), (868.0, 28.5), (866.5, 29.0), (866.0, 29.5), (866.0, 30.0), (865.5, 31.0), (865.0, 31.5), (865.0, 32.0), (865.5, 33.0), (864.5, 34.0), (863.0, 35.0), (863.5, 36.0), (866.5, 36.5), (868.0, 37.0), (866.5, 37.5), (865.0, 38.0), (864.0, 39.0), (863.5, 40.0), (862.5, 40.5), (861.0, 41.5), (859.5, 42.5), (857.0, 44.0), (856.5, 45.0), (855.0, 45.5), (854.5, 46.5), (853.5, 47.5), (852.0, 48.5), (850.5, 50.0), (849.5, 51.0), (848.0, 52.0), (846.5, 53.0), (844.5, 53.5), (843.5, 54.0), (842.0, 55.0), (841.0, 56.0), (839.5, 57.0), (837.5, 58.5), (836.0, 60.0), (834.5, 61.0), (832.5, 62.5), (830.5, 63.5), (828.5, 66.0), (826.5, 66.5), (825.5, 67.5), (823.5, 68.5), (821.0, 69.0), (819.0, 70.5), (817.0, 71.5), (815.0, 72.5), (813.5, 74.0), (811.0, 75.5), (810.5, 77.5), (808.5, 79.0), (806.0, 82.0), (804.0, 85.5), (802.5, 88.5), (800.5, 91.5), (798.5, 93.5), (795.5, 95.5), (794.5, 98.0), (791.5, 101.0), (790.0, 103.5), (787.5, 106.0), (785.0, 108.5), (783.0, 111.0), (781.0, 113.5), (780.0, 117.0), (777.0, 120.0), (774.0, 123.0), (771.0, 125.0), (768.5, 128.0), (765.0, 131.0), (763.0, 133.5), (761.5, 136.5), (759.5, 139.0), (757.5, 142.0), (755.5, 144.5), (754.0, 147.5), (751.0, 151.0), (748.0, 154.0), (747.5, 157.5), (744.5, 160.0), (742.5, 162.5), (740.0, 166.5), (737.0, 170.0), (733.5, 173.0), (730.0, 176.5), (728.0, 179.5), (725.0, 183.5), (722.5, 186.5), (720.5, 189.5), (718.0, 192.0), (714.5, 195.5), (711.5, 198.5), (707.5, 202.5), (705.0, 207.0), (702.5, 210.5), (700.5, 214.5), (696.5, 218.0), (694.5, 221.5), (691.5, 225.0), (688.5, 230.0), (685.0, 233.5), (682.0, 237.5), (678.5, 241.0), (675.5, 245.5), (672.0, 249.5), (668.5, 253.5), (665.0, 258.0), (661.0, 265.0), (659.0, 268.0), (654.5, 272.5), (651.0, 276.0), (647.0, 279.5), (643.5, 284.0), (639.5, 288.5), (635.5, 293.5), (631.0, 298.5), (628.0, 304.5), (625.0, 309.5), (621.5, 314.0), (616.5, 318.5), (613.0, 322.5), (608.5, 328.0), (605.0, 334.0), (601.5, 339.0), (598.0, 343.5), (593.0, 347.5), (589.5, 353.0), (584.5, 358.0), (579.0, 366.0), (574.0, 371.5), (571.5, 376.0), (567.0, 381.0), (561.5, 386.5), (558.5, 392.5), (553.5, 398.0), (547.5, 404.5), (543.0, 410.5), (539.5, 417.0), (534.5, 423.5), (529.0, 428.0), (525.0, 433.5), (520.0, 441.0), (514.0, 447.0), (509.5, 454.0), (504.5, 460.0), (499.0, 465.5), (494.0, 472.5), (489.0, 480.0), (483.5, 486.5), (478.0, 494.0), (473.0, 500.5), (467.5, 508.5), (461.5, 515.5), (455.0, 521.5), (450.5, 529.5), (444.5, 537.0), (438.0, 545.0), (432.0, 552.0), (426.0, 559.5), (420.5, 567.5), (414.5, 576.0), (408.0, 583.0), (402.0, 591.0), (395.0, 599.5), (389.5, 607.5), (383.0, 614.5), (377.0, 619.5), (370.0, 623.0), (364.0, 627.5), (356.5, 631.5), (350.0, 636.0), (346.0, 640.0), (341.0, 644.5), (338.5, 649.5), (333.5, 653.0), (330.5, 657.5), (325.0, 662.0), (322.5, 666.0), (317.5, 670.5), (313.0, 675.5), (309.5, 681.0), (305.5, 686.0), (303.5, 690.5), (299.5, 695.5)]
    ns_path_vectors = np.diff(np.array(ns_path), axis=0) # compute differences between consecutive points (accounts for parallel paths)

    ns_path_ped_left = [(1010.0, 449.0), (1009.5, 449.5), (1008.5, 450.5), (413.0, 240.5), (377.0, 228.0), (407.5, 238.0), (405.5, 237.5), (405.0, 238.0), (403.5, 238.0), (403.0, 239.5), (402.0, 240.0), (401.0, 241.0), (400.0, 242.0), (399.0, 243.0), (400.5, 245.5), (400.0, 247.5), (398.5, 250.0), (397.5, 252.5), (396.5, 253.0), (396.0, 254.5), (395.5, 255.5), (395.5, 256.5), (394.5, 257.5), (394.0, 257.0), (393.5, 257.0), (392.0, 258.0), (390.5, 258.0), (389.5, 257.5), (387.5, 257.0), (386.5, 259.0), (386.0, 260.5), (386.0, 263.0), (387.0, 267.0), (385.0, 269.0), (383.0, 268.5), (382.5, 269.5), (381.0, 270.5), (380.5, 271.5), (379.5, 271.5), (378.5, 271.0), (377.0, 270.0), (377.0, 269.0), (374.5, 270.5), (373.0, 270.5), (371.5, 271.0), (370.0, 272.0), (369.0, 273.0), (367.0, 275.0), (359.5, 274.0), (357.0, 274.5), (375.0, 288.0), (368.5, 287.0), (372.5, 290.5), (371.5, 291.0), (369.5, 291.0), (368.0, 290.5), (370.0, 292.5), (370.0, 293.5), (369.0, 294.5), (367.5, 294.0), (358.0, 290.5), (358.0, 292.0), (357.5, 292.5), (356.5, 293.0), (358.0, 295.0), (360.0, 297.0), (360.5, 300.5), (359.0, 302.5), (338.0, 291.0), (348.5, 299.0), (347.5, 300.0), (350.0, 304.5), (352.0, 306.5), (353.0, 308.5), (352.5, 308.0), (344.5, 303.0), (340.0, 301.5), (339.0, 301.5), (337.5, 302.0), (344.0, 312.0), (343.0, 314.5), (342.5, 316.0), (333.5, 328.5), (333.0, 330.0), (331.0, 330.0), (331.5, 330.5), (330.0, 331.5), (330.5, 332.0), (331.0, 334.5), (330.5, 336.5), (330.0, 338.0), (319.5, 338.5), (327.0, 340.0), (326.5, 341.0), (311.0, 340.5), (309.5, 343.0), (308.0, 344.0), (303.5, 345.0), (302.0, 345.5), (307.5, 350.5), (307.0, 353.5), (305.0, 357.0), (306.5, 359.5), (305.5, 361.5), (305.0, 363.0), (305.0, 365.0), (303.5, 366.0), (302.5, 367.0), (301.5, 367.5), (300.5, 367.5), (294.0, 367.0), (291.0, 365.5), (293.5, 368.5), (292.5, 369.5), (286.5, 367.0), (285.0, 367.5), (283.0, 374.5), (282.5, 375.0), (274.5, 371.0), (278.0, 373.5), (278.5, 374.5), (279.0, 377.5), (277.5, 378.0), (276.5, 379.0), (274.5, 378.5), (273.5, 378.5), (272.5, 379.0), (271.5, 382.0), (270.5, 383.0), (267.5, 385.0), (268.0, 389.5), (267.5, 393.5), (268.0, 395.5), (267.0, 397.5), (271.5, 401.0), (268.5, 402.0), (266.0, 401.0), (265.5, 402.5), (261.5, 400.0), (261.0, 400.0), (259.0, 401.0), (258.5, 400.0), (257.5, 400.5), (296.0, 436.0), (298.5, 438.0), (283.0, 445.0), (283.5, 446.5), (284.0, 449.5), (282.0, 452.5), (279.5, 455.5), (279.0, 457.5), (281.0, 460.5), (281.0, 462.5), (226.5, 438.5), (226.0, 439.0), (221.5, 438.0), (223.5, 440.5), (222.5, 442.0), (222.5, 444.5), (221.5, 448.0), (220.5, 450.0), (220.0, 452.5), (219.0, 453.0), (217.0, 454.0), (220.5, 457.5), (218.0, 455.0), (215.5, 456.0), (220.5, 460.5), (226.5, 471.5), (224.5, 471.0), (223.5, 474.0), (223.5, 476.0), (188.0, 490.5), (191.0, 492.5), (189.0, 495.5), (188.0, 496.5), (176.0, 497.5), (174.5, 498.5), (173.0, 499.0), (172.0, 503.0), (171.5, 505.0), (170.0, 506.5), (169.5, 514.0), (168.0, 513.0), (167.5, 514.5), (166.5, 515.5), (154.0, 528.0), (155.0, 541.5), (153.5, 543.5), (152.0, 547.0), (151.5, 549.5), (151.0, 547.0), (150.0, 545.5), (150.0, 550.0), (146.5, 549.0), (145.5, 548.0), (146.5, 552.0), (145.5, 554.0), (332.0, 70.5), (341.0, 48.0), (198.0, 44.5), (198.0, 43.5), (199.5, 44.5), (200.0, 45.0), (200.5, 45.0), (202.5, 52.5), (203.5, 54.0), (203.0, 55.5), (210.5, 56.5), (211.0, 56.5), (211.5, 56.5), (213.0, 58.0), (214.5, 59.5), (215.0, 60.0), (277.5, 46.5), (276.5, 45.5), (275.0, 44.5), (274.0, 44.0), (278.5, 43.5), (272.0, 44.0), (271.0, 44.0), (269.5, 43.5), (268.5, 43.5), (270.5, 43.0)]
    ns_path_ped_right = [(837.5, 425.5), (837.5, 427.0), (837.5, 427.5), (837.5, 429.0), (837.5, 430.5), (837.0, 430.5), (837.5, 432.5), (837.5, 433.5), (1034.0, 393.0), (1052.0, 390.0), (1030.0, 403.5), (1030.0, 405.5), (1029.0, 409.0), (1029.0, 410.0), (1029.5, 411.5), (1028.5, 412.0), (1028.0, 414.0), (1027.0, 414.0), (1031.5, 414.5), (1002.0, 430.5), (1005.5, 432.5), (1007.5, 435.0), (1003.0, 435.5), (1001.5, 436.5), (999.0, 437.5), (1014.0, 438.5), (1013.0, 439.0), (1012.5, 439.5), (1013.0, 440.5), (1001.5, 453.0), (999.5, 457.0), (999.0, 458.0), (997.0, 459.5), (997.0, 460.5), (996.0, 461.5), (997.5, 462.5), (997.0, 463.0), (996.5, 464.5), (979.5, 468.5), (995.0, 470.0), (981.5, 475.0), (982.0, 476.0), (982.0, 478.5), (980.5, 480.5), (979.5, 481.0), (978.0, 482.5), (975.5, 483.5), (974.5, 485.0), (975.0, 484.5), (976.0, 485.0), (975.5, 485.5), (977.5, 488.5), (976.0, 490.5), (975.5, 491.5), (970.0, 512.0), (969.5, 513.5), (964.5, 518.5), (955.0, 521.5), (959.0, 525.0), (960.0, 527.0), (952.0, 528.0), (950.5, 530.0), (952.0, 530.0), (948.5, 531.0), (947.0, 532.5), (956.0, 533.5), (955.0, 534.0), (953.5, 535.5), (953.5, 536.0), (953.0, 537.0), (929.5, 570.0), (929.5, 572.0), (928.5, 573.5), (917.0, 589.0), (916.5, 590.5), (912.0, 600.0), (912.0, 602.0), (911.5, 603.0), (912.5, 604.0), (911.5, 604.5), (911.5, 605.5), (906.5, 604.5), (905.5, 605.0), (904.5, 606.0), (902.0, 607.5), (900.5, 608.5), (897.5, 612.0), (894.0, 613.5), (894.0, 618.0), (893.0, 619.5), (893.0, 620.5), (892.5, 622.0), (891.5, 623.5), (891.0, 623.5), (889.5, 624.5), (888.0, 626.0), (881.0, 634.5), (879.5, 638.0), (878.0, 642.0), (877.5, 643.5), (876.5, 645.0), (876.0, 646.5), (875.5, 647.5), (874.5, 649.0), (874.5, 652.0), (873.5, 652.5), (873.0, 654.0), (864.5, 656.0), (862.5, 659.5), (859.5, 662.5), (859.5, 666.5), (858.0, 668.5), (858.0, 669.0), (857.5, 671.0), (857.0, 672.5), (856.0, 674.0), (848.0, 689.0), (847.5, 691.0), (846.5, 692.0), (845.5, 693.0), (845.0, 694.0), (845.0, 694.5), (844.0, 695.5), (844.0, 696.5), (843.5, 697.5), (837.5, 702.5), (836.5, 703.5)]
    ew_path_ped_up = [(986.5, 479.5), (985.5, 483.5), (984.0, 485.0), (983.0, 487.5), (978.5, 492.5), (978.0, 494.0), (970.0, 509.5), (969.5, 511.0), (969.5, 513.0), (968.5, 515.0), (873.0, 309.5), (871.5, 310.0), (869.0, 306.5), (868.0, 306.0), (866.5, 304.0), (865.0, 303.5), (863.5, 304.0), (860.5, 304.0), (859.5, 303.5), (858.5, 303.0), (857.5, 303.0), (857.5, 303.5), (858.0, 304.0), (857.5, 302.5), (857.0, 302.0), (855.5, 301.5), (853.5, 302.0), (853.0, 301.5), (851.0, 301.5), (849.0, 300.5), (847.5, 299.5), (844.5, 298.5), (844.0, 297.5), (844.0, 297.0), (842.5, 297.0), (841.5, 297.5), (840.0, 296.5), (839.0, 296.0), (838.5, 295.5), (837.5, 295.0), (837.0, 296.0), (835.5, 294.5), (834.5, 293.5), (833.5, 294.0), (832.0, 291.5), (830.5, 291.5), (830.0, 290.0), (829.0, 290.0), (829.0, 288.0), (828.5, 287.5), (827.5, 286.5), (827.0, 286.5), (826.0, 286.0), (825.5, 286.0), (825.0, 285.5), (824.0, 285.5), (823.0, 285.5), (821.5, 285.5), (820.0, 285.0), (819.0, 284.5), (818.0, 285.0), (817.5, 285.0), (816.5, 284.0), (816.0, 284.5), (815.5, 284.0), (815.0, 283.5), (814.5, 284.0), (814.5, 283.5), (814.0, 283.5), (813.5, 283.0), (814.0, 282.5), (813.5, 282.5), (813.5, 282.0), (813.0, 281.5), (813.0, 281.0), (813.5, 281.0), (813.5, 280.5), (813.5, 279.5), (813.0, 279.5), (812.0, 279.0), (812.0, 278.5), (811.0, 279.0), (810.5, 278.5), (810.5, 278.0), (810.0, 277.5), (809.5, 277.0), (809.5, 276.0), (809.0, 274.5), (808.5, 274.0), (808.0, 273.0), (807.5, 272.5), (807.0, 272.5), (807.0, 273.0), (806.5, 273.0), (806.0, 272.0), (805.5, 272.0), (805.0, 272.0), (805.0, 272.5), (804.0, 272.5), (803.5, 272.0), (803.0, 271.5), (802.5, 272.5), (802.0, 272.0), (801.5, 272.5), (801.0, 272.0), (800.5, 272.0), (800.0, 272.0), (799.0, 272.0), (799.0, 271.5), (798.0, 271.0), (798.0, 270.5), (798.0, 269.5), (798.0, 269.0), (798.0, 268.5), (798.0, 268.0), (798.0, 267.5), (797.0, 267.5), (796.0, 267.0), (795.0, 265.0), (794.5, 265.5), (793.5, 264.5), (791.5, 261.5), (791.0, 260.5), (791.0, 260.0), (789.5, 259.0), (788.0, 257.0), (787.0, 255.5), (786.0, 254.0), (784.5, 252.5), (784.5, 254.5), (784.5, 253.0), (784.5, 252.0), (783.5, 251.5), (782.5, 252.0), (781.5, 251.0), (780.0, 251.5), (778.0, 251.0), (778.0, 250.5), (777.5, 249.5), (765.0, 250.0)]


    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print image dimensions
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add object count and name to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
    
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]


                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            cx = (output[0] + output[2]) / 2
                            cy = (output[1] + output[3]) / 2

                            # if(id==10):
                            #     we_path.append((cx, cy))
                            # if(id==12):
                            #     ns_path.append((cx, cy))

                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                
                # if (frame_idx + 1) % 10 == 0:
                    # LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)') # print s and timing info every 10 frames
                


            else:
                strongsort_list[i].increment_ages()
                LOGGER.info('No detections')

            # Display crossing information on the video
            font        = cv2.FONT_HERSHEY_SIMPLEX
            font_scale  = 0.5
            thickness   = 1
            padding     = 2    # px of padding around the text
            bg_color    = (0, 0, 0)
            text_color  = (0, 0, 255)
            while len(cross_display) > 5:  # ensure we have at least 5 lines to display
                cross_display.pop(0)  # remove the oldest line if we have more than 5
            for j, text in enumerate(cross_display):
                (w, h), base = cv2.getTextSize(text, font, font_scale, thickness)
                x, y = 10, 30 + j * 30
                tl = (x - padding, y - h - base - padding)
                br = (x + w + padding, y + base + padding)
                cv2.rectangle(im0, tl, br, (255, 255, 255), cv2.FILLED)
                cv2.putText(
                    im0,                  # draw onto this image
                    text,                 # the string to draw
                    (x, y),              # bottom‐left corner of text (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,           # font scale
                    text_color,          # color = red (B, G, R)
                    thickness,            # line thickness
                    cv2.LINE_AA
                )
            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

            # ——— track centers each frame ———
            current_ids = set()

            # If we have any detections for this source:
            if outputs[i] is not None and len(outputs[i]) > 0:
                for out in outputs[i]:
                    tid = int(out[4])
                    cls_index=int(out[5])
                    cls_name= names[cls_index] if cls_index < len(names) else 'unknown'
                    if tid not in id_to_class: # remember class on first frame detection
                        id_to_class[tid] = cls_name  # map track ID to class label
                    # compute center
                    cx = (out[0] + out[2]) / 2
                    cy = (out[1] + out[3]) / 2

                    # init per‐ID storage
                    if tid not in coords:
                        coords[tid] = []

                    if (cx.item(), cy.item()) not in coords[tid]:
                        coords[tid].append((cx.item(), cy.item())) # only add if not already present (e.g. if car is stopped)
                    current_ids.add(tid)

            # detect IDs that just disappeared
            lost = previous_ids - current_ids
            previous_ids = current_ids
            ids_per_frame.append(current_ids)  # store IDs seen in this frame

            # count how many objects crossed from west to east
            
            for tid, points in coords.items():
                # if tid in lost and (id_to_class[tid]=="person" and len(points) > 50 or id_to_class[tid]=="car" and len(points) > 20):  # if the track has disappeared
                recent_ids = ids_per_frame[(max(0, frame_idx - 10)):frame_idx]  # get IDs seen in the last 10 frames

                if not any(tid in frame_ids for frame_ids in recent_ids):  # if the track has disappeared in the last 5 frames (aka permanently)

                    # if id_to_class[tid] == "person" and len(points) > 100:
                    #     lcss_ew_up = lcss(ew_path_ped_up, points, eps=15.0)  # compute LCSS for east to west path
                    #     lcss_we_up = lcss(list(reversed(ew_path_ped_up)), points, eps=15.0)  # compute LCSS for west to east path
                    #     lcss_ns_left  = lcss(ns_path_ped_left, points, eps=15.0)  # compute LCSS for north to south path left
                    #     lcss_ns_right  = lcss(ns_path_ped_right, points, eps=15.0)  # compute LCSS for north to south path right
                    #     lcss_sn_left  = lcss(list(reversed(ns_path_ped_left)), points, eps=15.0)  # compute LCSS for south to north path left
                    #     lcss_sn_right  = lcss(list(reversed(ns_path_ped_right)), points, eps=15.0)  # compute LCSS for south to north path right

                    #     direction_map = {
                    #     lcss_ew_up:    "E-->W",
                    #     lcss_we_up:    "W-->E",
                    #     lcss_ns_left:  "N-->S",
                    #     lcss_ns_right: "N-->S",
                    #     lcss_sn_left:  "S-->N",
                    #     lcss_sn_right: "S-->N",
                    #     }

                    #     # determine path taken
                    #     max_lcss = max(lcss_ew_up, lcss_we_up, lcss_ns_left, lcss_ns_right, lcss_sn_left, lcss_sn_right)
                    #     if max_lcss > 0.5:  # if path is a match
                    #         path_taken = direction_map.get(max_lcss, "unknown") # match lcss to string directio
                    #     # path_taken = "E-->W" if lcss_ew_up > 0.5 and lcss_ew_up == max(lcss_ew_up, lcss_ns_left, lcss_ns_right) else "N-->S" if lcss_ns_left > 0.5 and lcss_ns_left == max(lcss_ew_up, lcss_ns_left, lcss_ns_right) else "unknown"
                    #         if not path_taken == "unknown" and tid not in crossed_ids:
                    #             count += 1
                    #             crossed_ids.add(tid)
                    #             ew.append(tid) if path_taken == "E-->W" else ns.append(tid) if path_taken == "N-->S" else we.append(tid) if path_taken == "W-->E" else sn.append(tid) if path_taken == "S-->N" else None
                    #             msg = f"{id_to_class[tid]} ({tid}) crossed {path_taken}"
                    #             cross_display.append(msg) # message to display on video
                    #             print(str(tid) + ' (' + id_to_class.get(tid, 'unknown') + f') crossed from {path_taken}')
                    #         print(f"{tid} ({id_to_class.get(tid, 'unknown')}) lcss_ew: {lcss_ew_up}, lcss_ns_left: {lcss_ns_left}, lcss_ns_right: {lcss_ns_right}, lcss_we: {lcss_we_up}, lcss_sn_left: {lcss_sn_left}, lcss_sn_right: {lcss_sn_right}\n")

                    if id_to_class[tid] == "car":
                        if len(points) > 20:  # if the track is long enough
                            points_vectors = np.diff(np.array(points), axis=0)         # compute differences between consecutive points (accounts for parallel paths)

                            distance_we, warp_path_we = fastdtw(points_vectors, we_path_vectors, dist=cosine)
                            distance_ns, warp_path_ns = fastdtw(points_vectors, ns_path_vectors, dist=cosine)
                            avg_we = distance_we / len(warp_path_we)
                            avg_ns = distance_ns / len(warp_path_ns)
                            # lcss_we = lcss(we_path_vectors, points_vectors, eps=10.0)  # compute LCSS  for west to east path vectors
                            # lcss_ns = lcss(ns_path_vectors, points_vectors, eps=10.0)  # compute LCSS for north to south path vectors
                            path_taken = "W-->E" if avg_we < avg_ns and avg_we < 0.3 else "N-->S" if avg_ns < avg_we and avg_ns < 0.3 else "unknown"
                            if not path_taken == "unknown" and tid not in crossed_ids:
                                count += 1
                                crossed_ids.add(tid)
                                we.append(tid) if path_taken == "W-->E" else ns.append(tid) if path_taken == "N-->S" else None
                                msg = f"{id_to_class[tid]} ({tid}) crossed {path_taken}"
                                cross_display.append(msg) # message to display on video
                                print(str(tid) + ' (' + id_to_class.get(tid, 'unknown') + f') crossed from {path_taken}')
                                print(f"{tid} ({id_to_class.get(tid, 'unknown')}) dtw_we: {avg_we}, dtw_ns: {avg_ns}+\n")
                   


    # ——— report ———
    print(f"{len(we)} objects crossed from west to east")
    print(f"{len(ew)} objects crossed from east to west")
    # print(f"{len(ne)} objects turned from north to east")
    print(f"{len(ns)} objects crossed from north to south")
    print(f"{len(sn)} objects turned from south to north")
    print(f"Tracked IDs (west to east): {[str(tid) + ' (' + id_to_class.get(tid, 'unknown') + ')' for tid in we]}")
    print(f"Tracked IDs (east to west): {[str(tid) + ' (' + id_to_class.get(tid, 'unknown') + ')' for tid in ew]}")
    # print(f"Tracked IDs (north to east): {[str(tid) + ' (' + id_to_class.get(tid, 'unknown') + ')' for tid in ne]}")
    print(f"Tracked IDs (north to south): {[str(tid) + ' (' + id_to_class.get(tid, 'unknown') + ')' for tid in ns]}")
    print(f"Tracked IDs (south to north): {[str(tid) + ' (' + id_to_class.get(tid, 'unknown') + ')' for tid in sn]}")

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)