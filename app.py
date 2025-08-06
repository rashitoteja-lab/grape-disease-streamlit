import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import json
from model import GrapeDiseaseCNN

# Load model and label encoding
@st.cache_resource
def load_model():
    with open("label_encoding.json", "r") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    model = GrapeDiseaseCNN(num_classes)
    model.load_state_dict(torch.load("grape_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model, label_map

model, label_map = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# UI
st.title("🍇 Grape Disease Classifier")
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)  # shape: (1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = list(label_map.keys())[list(label_map.values()).index(predicted.item())]

    st.success(f"✅ Predicted Disease: **{predicted_class}**")
