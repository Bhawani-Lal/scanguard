import numpy as np
import streamlit as st
from PIL import Image
from predict import load_model, load_threshold, predict

st.set_page_config(page_title="ScanGuard", page_icon="🔍", layout="centered")

# Probabilities in this band are too close to call: send to a human
REVIEW_BAND = (0.20, 0.60)


@st.cache_resource
def get_model_and_threshold():
    # Loaded once, then reused for every upload
    return load_model(), load_threshold()


def looks_like_casting(image):
    """Training images are greyscale. A very colourful image is probably not a casting photo."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    colourfulness = np.mean(np.abs(arr[..., 0] - arr[..., 1]) + np.abs(arr[..., 1] - arr[..., 2]))
    return colourfulness < 15


model, threshold = get_model_and_threshold()

st.title("ScanGuard")
st.write("Upload a top-view photo of a cast pump impeller. ScanGuard classifies it as "
         "**defective** or **OK** using a ResNet18 model tuned to minimise missed defects.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", width=300)

    if not looks_like_casting(image):
        st.warning("This doesn't look like a casting inspection photo. ScanGuard was trained "
                   "only on greyscale images of cast impellers, so any result would be unreliable.")
    else:
        label, p_def = predict(image, model, threshold)

        if label == "DEFECTIVE":
            st.error(f"DEFECTIVE: reject this part (defect probability {p_def:.1%})")
        else:
            st.success(f"OK: part passes inspection (defect probability {p_def:.1%})")

        st.progress(p_def, text=f"Defect probability: {p_def:.1%}  |  threshold: {threshold:.0%}")

        if REVIEW_BAND[0] <= p_def <= REVIEW_BAND[1]:
            st.info("Borderline result: send this part for manual review by an inspector.")

with st.expander("About this model"):
    st.markdown(
        "- **Model:** ResNet18, pretrained on ImageNet, final layer retrained (transfer learning)\n"
        "- **Data:** 7,348 casting images (public Kaggle dataset)\n"
        "- **Threshold:** 0.35, chosen on a held-out validation set to prioritise catching defects\n"
        "- **Test results:** 99.2% accuracy, 99.6% precision, 99.1% recall (4 missed defects out of 453)\n"
        "- **Limitation:** works only on this product type and camera setup"
    )