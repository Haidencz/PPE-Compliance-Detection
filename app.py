from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO


MODEL_PATH = Path("models/ppe_best.pt")


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


st.set_page_config(
    page_title="PPE Compliance Detection",
    page_icon="🦺",
    layout="wide"
)

st.title("PPE Compliance Detection")

st.write(
    "Upload a construction-site image to detect workers, "
    "personal protective equipment and possible safety violations."
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model()

confidence_threshold = st.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05
)

uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")

    st.subheader("Original image")
    st.image(image, use_container_width=True)

    if st.button("Run PPE detection", type="primary"):
        with st.spinner("Analysing image..."):
            results = model.predict(
                source=np.array(image),
                conf=confidence_threshold
            )

            result = results[0]

            # Ultralytics returns a BGR image, so convert it to RGB.
            annotated_image = result.plot()[:, :, ::-1]

        st.subheader("Detection result")
        st.image(annotated_image, use_container_width=True)

        detected_classes = [
            model.names[int(box.cls.item())]
            for box in result.boxes
        ]

        detection_counts = Counter(detected_classes)

        st.subheader("Detection summary")

        if detection_counts:
            for class_name, count in detection_counts.most_common():
                st.write(f"**{class_name}:** {count}")
        else:
            st.info("No objects were detected at this confidence threshold.")

        violation_classes = {
            "NO-Hardhat",
            "NO-Mask",
            "NO-Safety Vest"
        }

        violation_count = sum(
            detection_counts[class_name]
            for class_name in violation_classes
        )

        st.subheader("Compliance result")

        if violation_count > 0:
            st.warning(
                f"{violation_count} possible PPE violation(s) detected."
            )
        else:
            st.success("No PPE violations detected.")