import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 設置環境變數以避免 OpenMP 衝突

from ultralytics.models.yolo.detect import DetectionTrainer

# 定義訓練參數
args = dict(
    model="yolo11m.pt",  # 預訓練模型路徑
    data="C:/Users/cheng/Desktop/python/defect_yolov8/ultralytics-main/data/brain-tumor.yaml",
    epochs=800,  # 訓練週期數
    imgsz=800,  # 圖片大小
    batch=16,  # 批次大小
    workers=4,  # 數據加載線程數
    device=0  # 使用 GPU（若無 GPU，設為 "cpu"）
)

# 實例化並運行訓練器

if __name__ == '__main__':
    trainer = DetectionTrainer(overrides=args)
    trainer.train()

    print("訓練完成！")