# app.py
import streamlit as st
from PIL import Image, ImageDraw
import torch
from torchvision import models, transforms
import json
import os

st.set_page_config(page_title="AI Fabric Laundry Sorter", page_icon="Laundry", layout="centered")
st.title("AI Fabric Laundry Sorter")
st.markdown("Upload **any photo** of your clothing — I'll **zoom in** and analyze the fabric texture.")

# ------------------- LOAD MODEL & METADATA -------------------
@st.cache_resource
def load_model_and_metadata():
    model_path = "best_convnext_tiny_fabric.pth"
    meta_path = "metadata.json"

    if not os.path.exists(model_path):
        st.error(f"Model not found: `{model_path}`")
        st.stop()
    if not os.path.exists(meta_path):
        st.error(f"Metadata not found: `{meta_path}`")
        st.stop()

    with open(meta_path) as f:
        meta = json.load(f)

    class_names = meta['class_names']
    washing = meta['washing_recommendations']

    model = models.convnext_tiny(pretrained=False)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=meta['normalize_mean'], std=meta['normalize_std'])
    ])

    return model, transform, class_names, washing

model, transform, class_names, washing_recommendations = load_model_and_metadata()

# ------------------- AUTO ZOOM FUNCTION -------------------
def auto_zoom_closeup(image, zoom_factor=0.5):
    """
    Crop center of image to simulate close-up.
    zoom_factor=0.5 → keep 50% center (zoomed in 2x)
    """
    w, h = image.size
    crop_w = int(w * zoom_factor)
    crop_h = int(h * zoom_factor)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h
    return image.crop((left, top, right, bottom))

# Add red border to show zoomed area
def draw_zoom_box(image, zoom_factor=0.5):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    crop_w = int(w * zoom_factor)
    crop_h = int(h * zoom_factor)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h
    draw.rectangle([left, top, right, bottom], outline="red", width=3)
    return image

# ------------------- INFERENCE -------------------
def predict(img):
    x = transform(img.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, 1)
    fabric = class_names[idx.item()]
    prob = conf.item() * 100
    rec = washing_recommendations.get(fabric, "No instructions.")
    return fabric, prob, rec

# ------------------- UI -------------------
uploaded = st.file_uploader("Upload clothing photo", type=["jpg", "jpeg", "png"])

if uploaded:
    original = Image.open(uploaded)
    
    # Show original with zoom box
    img_with_box = original.copy()
    img_with_box = draw_zoom_box(img_with_box, zoom_factor=0.5)
    st.image(img_with_box, caption="Original + Zoom Area (Red Box)", use_column_width=True)

    # Auto zoom
    zoomed = auto_zoom_closeup(original, zoom_factor=0.5)
    st.image(zoomed, caption="Analyzing This Close-Up", use_column_width=True)

    with st.spinner("Analyzing fabric texture..."):
        fabric, prob, rec = predict(zoomed)

    st.success("Analysis Complete!")
    st.markdown(f"### **{fabric}**")
    st.markdown(f"**Confidence**: `{prob:.1f}%`")

    if prob < 70:
        st.warning("Low confidence – try a clearer or closer photo.")

    st.info(f"**Laundry Instructions**:\n\n{rec}")

    with st.expander("Top 3 Predictions"):
        with torch.no_grad():
            logits = model(transform(zoomed.convert('RGB')).unsqueeze(0))
            probs = torch.softmax(logits, dim=1).squeeze()
            topk = torch.topk(probs, 3)
            for i in range(3):
                f = class_names[topk.indices[i]]
                p = topk.values[i].item() * 100
                st.write(f"- **{f}**: {p:.1f}%")

st.markdown("---")
st.caption("ConvNeXt Tiny • 90.9% Accuracy • Auto Zoom for Better Texture Detection")