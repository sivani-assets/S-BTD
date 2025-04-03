import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from PIL import Image
from reportlab.pdfgen import canvas
import os

# Generate Grad-CAM
def generate_gradcam(model, img_path, class_index, save_dir):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    img = np.expand_dims(img / 255.0, axis=0)

    grad_model = keras.models.Model(inputs=model.input, outputs=[model.get_layer("conv2d_2").output, model.output])
    conv_output, predictions = grad_model.predict(img)
    weights = predictions[0, class_index]

    gradcam = np.dot(conv_output[0], weights)
    gradcam = np.maximum(gradcam, 0)
    gradcam = cv2.resize(gradcam, (256, 256))

    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    gradcam_path = os.path.join(save_dir, "gradcam.jpg")
    cv2.imwrite(gradcam_path, superimposed_img)
    return gradcam_path

# Generate CNN Feature Maps
def generate_feature_maps(model, img_path, save_dir):
    feature_map_model = keras.models.Model(inputs=model.input, outputs=model.get_layer("conv2d_1").output)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    img = np.expand_dims(img / 255.0, axis=0)

    feature_maps = feature_map_model.predict(img)
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(feature_maps[0, :, :, i], cmap="viridis")
        ax.axis("off")

    feature_map_path = os.path.join(save_dir, "feature_map.jpg")
    plt.savefig(feature_map_path)
    return feature_map_path

# Generate PDF Report
def generate_pdf_report(data):
    pdf_path = "static/reports/patient_report.pdf"
    c = canvas.Canvas(pdf_path)
    c.drawString(100, 800, f"Patient Name: {data['name']}")
    c.drawString(100, 780, f"Age: {data['age']}")
    c.drawString(100, 760, f"ID: {data['id']}")
    c.drawString(100, 740, f"Tumor Classification: {data['result']}")
    c.drawString(100, 720, f"Confidence Score: {data['confidence']}")
    c.drawImage(data["gradcam"], 100, 500, width=200, height=200)
    c.save()
    return pdf_path
