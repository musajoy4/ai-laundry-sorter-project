# app.py
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import json
import os

# -------------------------------
# 1. Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Fabric Laundry Sorter",
    page_icon="Laundry",
    layout="centered"
)

st.title("AI Fabric Laundry Sorter")
st.markdown("Upload a **close-up photo** of your clothing fabric. I'll tell you the **material** and **how to wash it safely**.")

# -------------------------------
# 2. Load Model & Metadata
# -------------------------------
@st.cache_resource
def load_model_and_metadata():
    # Paths
    model_path = "fabric_laundry_sorter_v1.pth"
    meta_path = "metadata.json"

    # Load metadata
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    class_names = meta['class_names']
    washing = meta['washing_recommendations']

    # Load model
    model = models.convnext_tiny(pretrained=False)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=meta.get('normalize_mean', [0.485, 0.456, 0.406]),
            std=meta.get('normalize_std', [0.229, 0.224, 0.225])
        )
    ])

    return model, transform, class_names, washing

model, transform, class_names, washing_recommendations = load_model_and_metadata()

# -------------------------------
# 3. Inference Function
# -------------------------------
def predict(image):
    img = image.convert('RGB')
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        confidence, idx = torch.max(probs, 1)

    fabric = class_names[idx.item()]
    prob = confidence.item() * 100
    recommendation = washing_recommendations.get(fabric, "No care instructions available.")

    return fabric, prob, recommendation

# -------------------------------
# 4. Upload & Predict
# -------------------------------
uploaded_file = st.file_uploader("Upload fabric photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Fabric", use_column_width=True)

    with st.spinner("Analyzing fabric..."):
        fabric, prob, rec = predict(image)

    st.success("Analysis Complete!")
    st.markdown(f"### This is **{fabric}**")
    st.markdown(f"**Confidence**: `{prob:.1f}%`")

    if prob < 70:
        st.warning("Low confidence. Consider a clearer or closer photo.")

    st.info(f"**Laundry Recommendation**:\n\n{rec}")

    # Optional: Show top 3 predictions
    with st.expander("Show top 3 predictions"):
        with torch.no_grad():
            logits = model(transform(image.convert('RGB')).unsqueeze(0))
            probs = torch.softmax(logits, dim=1).squeeze()
            top3_prob, top3_idx = torch.topk(probs, 3)
            for i in range(3):
                f = class_names[top3_idx[i]]
                p = top3_prob[i].item() * 100
                st.write(f"- **{f}**: {p:.1f}%")

# -------------------------------
# 5. Footer
# -------------------------------
st.markdown("---")
st.caption("Built with ConvNeXt Tiny • 90.9% Test Accuracy • 25 Fabric Classes")