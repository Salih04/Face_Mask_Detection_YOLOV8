import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import shutil
import torch
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import LabelEncoder
import wandb
from ultralytics import YOLO

def main():
    # Log in to wandb with API key
    wandb.login(key='eb4c4a1fa7eec1ffbabc36420ba1166f797d4ac5')

    ANNOTATIONS_DIR = "C:\\Users\\salih\\Desktop\\Face_Mask_Detection_using_YOLOv8m\\annotations"
    IMAGES_DIR = "C:\\Users\\salih\\Desktop\\Face_Mask_Detection_using_YOLOv8m\\images"
    OUTPUT_DIR = "C:\\Users\\salih\\Desktop\\Face_Mask_Detection_using_YOLOv8m\\output"

    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'val'), exist_ok=True)

    annotations = []
    for xml_file in os.listdir(ANNOTATIONS_DIR):
        tree = ET.parse(os.path.join(ANNOTATIONS_DIR, xml_file))
        root = tree.getroot()
        file_name = root.find('filename').text
        objects = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            objects.append((class_name, xmin, ymin, xmax, ymax))
        annotations.append((file_name, objects))

    # Split dataset into training and validation sets
    train_annotations = annotations[:int(0.8*len(annotations))]
    val_annotations = annotations[int(0.8*len(annotations)):]

    # Encode labels
    label_encoder = LabelEncoder()
    all_labels = [obj[0] for ann in annotations for obj in ann[1]]
    label_encoder.fit(all_labels)

    # Helper function to convert annotations to YOLO format
    def convert_to_yolo_format(size, box):
        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[2]) / 2.0
        y = (box[1] + box[3]) / 2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x, y, w, h)

    # Save images and labels in YOLO format
    def save_yolo_files(annotations, image_dir, label_dir):
        for file_name, objects in annotations:
            img_path = os.path.join(IMAGES_DIR, file_name)
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            shutil.copy(img_path, image_dir)
            label_path = os.path.join(label_dir, file_name.replace('.png', '.txt'))
            with open(label_path, 'w') as f:
                for obj in objects:
                    class_name, xmin, ymin, xmax, ymax = obj
                    class_id = label_encoder.transform([class_name])[0]
                    bb = convert_to_yolo_format((w, h), (xmin, ymin, xmax, ymax))
                    f.write(f"{class_id} {' '.join(map(str, bb))}\n")

    # Save train and val data
    save_yolo_files(train_annotations, os.path.join(OUTPUT_DIR, 'images', 'train'), os.path.join(OUTPUT_DIR, 'labels', 'train'))
    save_yolo_files(val_annotations, os.path.join(OUTPUT_DIR, 'images', 'val'), os.path.join(OUTPUT_DIR, 'labels', 'val'))

    # Create data.yaml file
    data_yaml = f"""train: {os.path.join(OUTPUT_DIR, 'images', 'train')} val: {os.path.join(OUTPUT_DIR, 'images', 'val')} nc: {len(label_encoder.classes_)} names: {label_encoder.classes_.tolist()}
    """

    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        f.write(data_yaml)

    model = YOLO('yolov8n.pt')

    model.train(
        data=os.path.join(OUTPUT_DIR, 'data.yaml'), 
        epochs=80, 
        imgsz=416, 
        batch=16, 
        cache=True
    )

    # Evaluate and visualize results
    model = YOLO('C:\\Users\\salih\\Desktop\\Face_Mask_Detection_using_YOLOv8m\\runs\\detect\\train5\\weights\\best.pt')

    def plot_sample_images(image_dir, model, num_images=5):
        images = [os.path.join(image_dir, img) for img in os.listdir(image_dir)][:num_images]
        for img_path in images:
            img = cv2.imread(img_path)
            results = model(img)
            result_img = results[0].plot()  
            plt.figure(figsize=(10, 10))
            plt.imshow(result_img)
            plt.axis('off')
            plt.show()

    plot_sample_images(os.path.join(OUTPUT_DIR, 'images', 'val'), model)


if __name__ == "__main__":
    main()
