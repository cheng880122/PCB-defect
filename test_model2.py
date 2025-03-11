from ultralytics import YOLO
import cv2
import numpy as np
import os
import glob

# 設置模型路徑
model_path = "C:/Users/cheng/Desktop/python/defect_yolov8/ultralytics-main/runs/detect/train41/weights/best.pt"

# 設置測試照片資料夾路徑
source_folder = "C:/Users/cheng/Desktop/python/defect_yolov8/test_image/"

# 載入模型
model = YOLO(model_path)

# 獲取資料夾中的所有圖像文件（假設為 jpg 格式）
image_paths = glob.glob(os.path.join(source_folder, "*.jpg"))

# 定義圖像切割函數
def split_image(image, rows=4, cols=4):
    height, width = image.shape[:2]
    block_height = height // rows
    block_width = width // cols
    blocks = []
    for i in range(rows):
        for j in range(cols):
            x_start = j * block_width
            y_start = i * block_height
            block = image[y_start:y_start + block_height, x_start:x_start + block_width]
            blocks.append((block, x_start, y_start))
    return blocks

# 處理每張圖像
for image_path in image_paths:
    # 載入原始圖像
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法載入圖像: {image_path}")
        continue

    # 切割圖像成 16 塊
    blocks = split_image(image, rows=4, cols=4)

    # 對每塊進行檢測
    all_boxes = []
    for idx, (block, x_start, y_start) in enumerate(blocks):
        # 為每個塊生成唯一的臨時文件路徑
        temp_path = f"temp_block_{os.path.basename(image_path).split('.')[0]}_{idx}.jpg"
        cv2.imwrite(temp_path, block)
        
        # 進行預測
        results = model.predict(
            source=temp_path,
            conf=0.25,  # 置信度閾值
            iou=0.7,   # IOU 閾值
            save=False
        )
        
        # 處理檢測結果
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                confidence = box.conf.item()
                # 局部座標
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                # 轉換到全局座標
                x_min_global = x_start + x_min
                y_min_global = y_start + y_min
                x_max_global = x_start + x_max
                y_max_global = y_start + y_max
                all_boxes.append({
                    'class_name': class_name,
                    'confidence': confidence,
                    'box': [x_min_global, y_min_global, x_max_global, y_max_global]
                })
        
        # 刪除臨時文件
        os.remove(temp_path)

    # 在原始圖像上繪製檢測框
    for box_info in all_boxes:
        class_name = box_info['class_name']
        confidence = box_info['confidence']
        x_min, y_min, x_max, y_max = map(int, box_info['box'])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存結果
    save_path = os.path.join(source_folder, f"{os.path.basename(image_path).split('.')[0]}_detected.jpg")
    cv2.imwrite(save_path, image)
    print(f"檢測結果已保存至: {save_path}")