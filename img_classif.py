import streamlit as st
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

# Load model with caching
@st.cache_resource
def load_model():
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.8)
    model.eval()
    return model, weights

model, weights = load_model()
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

def make_prediction(img):
    with torch.no_grad():  # Disable gradient tracking
        img_preprocessed = img_preprocess(img)
        prediction = model(img_preprocessed.unsqueeze(0))
        prediction = prediction[0]
        prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_boxes(img, prediction):
    # Convert to PIL Image
    img_with_boxes = img.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    for box, label in zip(prediction["boxes"], prediction["labels"]):
        color = "red" if label == "person" else "green"
        draw.rectangle(box.tolist(), outline=color, width=2)
        draw.text((box[0], box[1]), label, fill=color)
    return img_with_boxes

st.title("Object Detector :star2: :earth_africa:")
upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])
if upload:
    img = Image.open(upload).convert("RGB")
    prediction = make_prediction(img)
    img_with_boxes = create_image_with_boxes(img, prediction)
    st.image(img_with_boxes, use_column_width=True)
    del prediction["boxes"]  # Remove boxes for display
    st.header("Predicted Probabilities")
    st.write(prediction)