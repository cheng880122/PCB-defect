from ultralytics import YOLO
import os

# 設置模型路徑
model_path = "C:/Users/cheng/Desktop/python/defect_yolov8/ultralytics-main/runs/detect/train41/weights/best.pt"

# 設置測試照片路徑（單張照片或目錄）
source_path = "C:/Users/cheng/Desktop/python/defect_yolov8/test_image/"  # 可改為單張照片，例如 "test.jpg"

# 載入模型
model = YOLO(model_path)

# 進行預測
results = model.predict(
    source=source_path,  # 測試照片或目錄
    conf=0.25,           # 置信度閾值（可調整，例如 0.5）
    iou=0.7,             # IoU 閾值（非極大值抑制）
    save=True,           # 保存預測結果
    save_txt=True,       # 保存標籤文件（.txt）
    save_conf=True,      # 保存置信度
    show=False           # 是否顯示結果（設為 False 以避免彈窗）
)

# 打印結果
for result in results:
    print(f"照片: {result.path}")
    print("檢測到的目標:")
    for box in result.boxes:
        class_id = int(box.cls)  # 類別 ID
        class_name = model.names[class_id]  # 類別名稱
        confidence = box.conf.item()  # 置信度
        coords = box.xyxy[0].tolist()  # 邊框坐標 [x_min, y_min, x_max, y_max]
        print(f"- 類別: {class_name}, 置信度: {confidence:.2f}, 坐標: {coords}")
    print("-" * 50)

# 結果保存路徑
print(f"預測結果已保存至: {results[0].save_dir}")

if __name__ == '__main__':
    pass  # 已包含主邏輯，無需額外代碼