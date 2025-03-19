from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import gradio as gr
import pandas as pd

trained_model_path = 'best.pt'  
model = YOLO(trained_model_path)

def inference_function(image):
    """
    Performs object detection inference on an input image using the loaded YOLOv8 model.
    Returns:
        tuple: (PIL Image with bounding boxes, Pandas DataFrame of detections)
                OR ("No objects detected", empty DataFrame) if no detections
    """
    pil_image = Image.fromarray(np.uint8(image)).convert('RGB')
    results_list = model(pil_image, verbose=False)
    results = results_list[0]

    detections = []
    image_np = np.array(pil_image)
    image_with_boxes_np = image_np.copy()

    if results.boxes:
        for *xyxy, conf, cls in results.boxes.data:
            x1, y1, x2, y2 = map(int, xyxy)
            class_id = int(cls)
            confidence = float(conf)
            class_name = model.names[class_id]

            detections.append({
                'class_name': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })

            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(image_with_boxes_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_with_boxes_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        detections_df = pd.DataFrame(detections)
        image_with_boxes_pil = Image.fromarray(image_with_boxes_np)
        return image_with_boxes_pil, detections_df
    else:
        return "No objects detected", pd.DataFrame()

# Gradio Interface
iface = gr.Interface(
    fn=inference_function,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(label="Detected Objects"),
        gr.DataFrame(label="Detections Table")
    ],
    examples=['3_jpg.rf.c46999f92d679a134503e2c481ca4d9d.jpg', '10_jpg.rf.438680c2fe8e5e96e41932b3276761e2.jpg', '100_jpg.rf.37be664a78a2a0e331cef5cb77186ca7.jpg'] 
)

iface.launch()