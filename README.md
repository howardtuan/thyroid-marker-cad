# Measurement-Marker Guided Segmentation &amp; Transfer-Learning Classification for Thyroid-US CAD
> 只要把一張 含“黃色測量標記” 的甲狀腺超音波影像丟進來，本系統就能：
> 1. 框出結節 ROI
> 2. 畫出結節遮罩 (Mask)
> 3. 給出良／惡性機率

> 全流程端到端，方便醫師即時驗證與輔助診斷。

# 1. 專案簡介
    1. YOLOv8-n：偵測測量標記外接的「結節方框」。
    1. U-Net++：在 ROI 內做結節像素級分割 (Mask)。
    1. ResNet-18：將 ROI+Mask 組合成三通道影像，輸出良性／惡性機率。

# 2. 資料夾結構

```csharp!
thyroid-marker-cad/
├─ data/
│  ├─ private/                        # 私有影像（皆含結節）
│  │   ├─ images_png/                 # 原圖 (去 DICOM tag、8-bit)
│  │   └─ yolo_labels/                # bbox 標註 txt：0 x y w h
│  └─ public/                         # 公開良/惡數據
│      ├─ tn3k/{images,masks}
│      └─ ddti/{images,masks}
├─ models/                            # 訓練產生的權重
│  ├─ yolov8n_marker.pt
│  ├─ unetpp_seg.pt
│  └─ resnet_cls.pt
├─ src/                               # 核心程式碼
│  ├─ 0_preprocess.py
│  ├─ 1_train_yolo.py
│  ├─ 2_generate_roi.py
│  ├─ 3_train_seg.py
│  ├─ 4_train_cls.py
│  ├─ 5_inference_pipeline.py
│  └─ utils.py
├─ notebooks/                         # EDA、測試草稿
├─ requirements.txt
└─ README.md
```

# 3. 環境安裝
```bash!
conda create -n thyus python=3.10 -y
conda activate thyus

# 依 CUDA 版本選擇對應 PyTorch，以下以 CUDA 12.x 為例
pip install torch==2.2.2+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121

# 主要依賴：ultralytics、segmentation_models_pytorch、timm、albumentations...
pip install -r requirements.txt
```

# 4. 資料準備
| 步驟| 說明|
| -------- | -------- | 
| 私有影像 PNG	     | 放入 `data/private/images_png/`     | 
| YOLO 標註	     | 每張圖畫**一個結節方框** → `data/private/yolo_labels/`，格式：`0 x_center y_center width height` (相對值 0~1)     | 
| 公開資料	     | Tex下載 TN3K、DDTI，放 `data/public/`；執行 `scripts/split_public.py` 會自動裁 ROI 並存對應 Mask     | 
# 5. 訓練流程
### 5.1. ROI 偵測（YOLOv8-n）
```bash!
python src/1_train_yolo.py \
  --data data/private/yolo.yaml \
  --epochs 50 --img 640
```
產物：`models/yolov8n_marker.pt`
> 目標指標：mAP@0.5 ≥ 0.90

### 5.2 依偵測結果裁 ROI
```bash!
python src/2_generate_roi.py \
  --weights models/yolov8n_marker.pt \
  --img_dir data/private/images_png \
  --out_dir data/private/roi_crops \
  --pad 0.1          # 四周留 10%
```
### 5.3 分割（U-Net++）
```bash!
python src/3_train_seg.py \
  --data_dir data/private/roi_crops \
  --mask_dir data/private/masks_manual \
  --epochs 80 --batch 8
```
產物：`models/unetpp_seg.pt`
>目標指標：Dice ≥ 0.80

### 5.4 良／惡分類（ResNet-18）
#### 1. 公開集預訓練
```bash!
python src/4_train_cls.py \
  --dataset public \
  --epochs 40 --batch 32 \
  --out models/resnet_pre.pt
```

#### 2.（可選）私有少量標籤微調
```bash!
python src/4_train_cls.py \
  --dataset private \
  --weights models/resnet_pre.pt \
  --freeze_layers 2 --epochs 10 \
  --out models/resnet_cls.pt
```
> 目標指標：AUC ≥ 0.85，F1 ≥ 0.80

# 6. 一鍵推論
```bash!
python src/5_inference_pipeline.py \
  --img path/to/sample.png \
  --det models/yolov8n_marker.pt \
  --seg models/unetpp_seg.pt \
  --cls models/resnet_cls.pt \
  --out result_vis.png
```
#### 輸出示範
```yaml!
Malignancy probability: 87.3 %
可視化結果已存 result_vis.png
```
# 7. 評估指令


| 模組	 | 指令| 指標 |
| -------- | -------- | -------- |
| 偵測     |   `python eval/eval_det.py	`     | mAP@0.5     |
| 分割	     | `python eval/eval_seg.py	`     | Dice / IoU     |
| 分類	     | `python eval/eval_cls.py	`     | ROC-AUC / F1 / AUPRC     |
| 端到端	     | `python eval/eval_end2end.py	`     | FPS、單張延遲     |

# 8. 引用格式
```bibex!
@misc{thyroid_marker_cad_2025,
  title  = {Measurement-Marker Guided Segmentation and Multi-Source Transfer Learning Classification for Thyroid Ultrasound},
  author = {Howard Tuan},
  year   = {2025},
  howpublished = {\url{https://github.com/howardtuan/thyroid-marker-cad}}
}
```
