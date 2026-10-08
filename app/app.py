# ============================================================
# Brain Tumor MRI Classification - Streamlit Application
# Final Model: ResNet50
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# CUSTOM RESNET50 PREPROCESSING LAYER
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="Custom"
)
class ResNet50Preprocess(tf.keras.layers.Layer):

    def call(self, inputs):
        return tf.keras.applications.resnet50.preprocess_input(
            inputs
        )


# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet50"
    / "best_resnet50.keras"
)


IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor"
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found:\n{MODEL_PATH}"
        )

    # Explicitly provide both possible names used by
    # Keras when serializing the custom layer.
    custom_objects = {
        "ResNet50Preprocess": ResNet50Preprocess,
        "Custom>ResNet50Preprocess": ResNet50Preprocess
    }

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False
    )

    return model


# ============================================================
# FIND RESNET50 BACKBONE
# ============================================================

def find_resnet_backbone(model):

    # Search direct child models
    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            if "resnet" in layer.name.lower():
                return layer

    # Search one level deeper
    for layer in model.layers:

        if hasattr(layer, "layers"):

            for sublayer in layer.layers:

                if isinstance(sublayer, tf.keras.Model):

                    if "resnet" in sublayer.name.lower():
                        return sublayer

    raise ValueError(
        "Could not find the ResNet50 backbone."
    )


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer(backbone):

    conv_layers = []

    for layer in backbone.layers:

        if isinstance(
            layer,
            (
                tf.keras.layers.Conv2D,
                tf.keras.layers.SeparableConv2D
            )
        ):

            conv_layers.append(layer)

    if not conv_layers:

        raise ValueError(
            "Could not find a convolutional layer."
        )

    return conv_layers[-1]


# ============================================================
# CREATE GRAD-CAM COMPONENTS
# ============================================================

@st.cache_resource
def create_gradcam_components(model):

    backbone = find_resnet_backbone(model)

    last_conv_layer = find_last_conv_layer(
        backbone
    )

    grad_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[
            last_conv_layer.output,
            backbone.output
        ],
        name="resnet50_gradcam_model"
    )

    backbone_index = model.layers.index(
        backbone
    )

    head_layers = model.layers[
        backbone_index + 1:
    ]

    return (
        backbone,
        last_conv_layer,
        grad_model,
        head_layers
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_batch = np.expand_dims(
        image_array,
        axis=0
    )

    return image_batch


# ============================================================
# PREDICTION
# ============================================================

def predict_image(model, image):

    image_batch = preprocess_image(
        image
    )

    predictions = model.predict(
        image_batch,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_class,
        confidence,
        probabilities
    )


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    image,
    grad_model,
    head_layers,
    predicted_index
):

    image = image.convert("RGB")

    resized_image = image.resize(
        IMG_SIZE
    )

    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    image_batch = np.expand_dims(
        image_array,
        axis=0
    )

    processed_image = (
        tf.keras.applications.resnet50.preprocess_input(
            image_batch
        )
    )

    with tf.GradientTape() as tape:

        conv_outputs, backbone_output = (
            grad_model(
                processed_image,
                training=False
            )
        )

        x = backbone_output

        for layer in head_layers:

            if isinstance(
                layer,
                tf.keras.layers.InputLayer
            ):
                continue

            x = layer(
                x,
                training=False
            )

        predictions = x

        class_score = predictions[
            :,
            predicted_index
        ]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:

        raise ValueError(
            "Grad-CAM gradients could not be calculated."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]

    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    return heatmap.numpy()


# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_heatmap_overlay(
    original_image,
    heatmap,
    alpha=0.45
):

    original_image = original_image.convert(
        "RGB"
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    heatmap_image = Image.fromarray(
        heatmap_uint8
    )

    heatmap_image = heatmap_image.resize(
        original_image.size
    )

    heatmap_array = np.array(
        heatmap_image
    )

    colored_heatmap = np.zeros(
        (
            heatmap_array.shape[0],
            heatmap_array.shape[1],
            3
        ),
        dtype=np.uint8
    )

    colored_heatmap[:, :, 0] = (
        heatmap_array
    )

    colored_heatmap[:, :, 1] = (
        heatmap_array * 0.4
    ).astype(np.uint8)

    colored_heatmap[:, :, 2] = (
        255 - heatmap_array
    )

    colored_heatmap = Image.fromarray(
        colored_heatmap
    )

    overlay = Image.blend(
        original_image,
        colored_heatmap,
        alpha
    )

    return overlay


# ============================================================
# MAIN APP
# ============================================================

def main():

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title(
        "🧠 Brain Tumor MRI Classifier"
    )

    st.markdown(
        """
        ### Deep Learning Based MRI Classification

        Upload a brain MRI image and classify it using the
        trained **ResNet50** deep learning model.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    with st.sidebar:

        st.header(
            "Model Information"
        )

        st.write(
            "**Model:** ResNet50"
        )

        st.write(
            "**Input Size:** 224 × 224"
        )

        st.write(
            "**Classes:** 4"
        )

        st.write(
            "**Test Accuracy:** 93.31%"
        )

        st.divider()

        st.subheader(
            "Classification Classes"
        )

        st.write("• Glioma")
        st.write("• Meningioma")
        st.write("• No Tumor")
        st.write("• Pituitary Tumor")

        st.divider()

        st.info(
            """
            This application is an educational and
            research prototype.

            It is not intended to replace professional
            medical diagnosis.
            """
        )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Loading ResNet50 model..."
        ):

            model = load_model()

    except Exception as e:

        st.error(
            "❌ ResNet50 model could not be loaded."
        )

        st.exception(e)

        st.stop()

    # --------------------------------------------------------
    # UPLOAD IMAGE
    # --------------------------------------------------------

    st.subheader(
        "Upload Brain MRI"
    )

    uploaded_file = st.file_uploader(
        "Choose an MRI image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is None:

        st.info(
            "Please upload an MRI image to begin."
        )

        return

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception as e:

        st.error(
            "Could not read the uploaded image."
        )

        st.exception(e)

        return

    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    st.subheader(
        "Uploaded MRI"
    )

    st.image(
        image,
        caption="Original MRI",
        width=500
    )

    st.divider()

    # --------------------------------------------------------
    # CLASSIFICATION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Classify MRI",
        type="primary",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        with st.spinner(
            "Analyzing MRI..."
        ):

            try:

                (
                    predicted_class,
                    confidence,
                    probabilities
                ) = predict_image(
                    model,
                    image
                )

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.exception(e)

                return

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.subheader(
            "Classification Result"
        )

        result_col1, result_col2 = st.columns(
            2
        )

        with result_col1:

            st.success(
                f"Prediction: "
                f"**{DISPLAY_NAMES[predicted_class]}**"
            )

        with result_col2:

            st.info(
                f"Confidence: "
                f"**{confidence * 100:.2f}%**"
            )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        st.subheader(
            "Class Probabilities"
        )

        for index, class_name in enumerate(
            CLASS_NAMES
        ):

            probability = (
                float(
                    probabilities[index]
                ) * 100
            )

            st.write(
                f"**{DISPLAY_NAMES[class_name]}:** "
                f"{probability:.2f}%"
            )

            st.progress(
                min(
                    max(
                        int(probability),
                        0
                    ),
                    100
                )
            )

        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "🔥 Grad-CAM Explainability"
        )

        st.write(
            """
            Grad-CAM highlights image regions that contributed
            to the model's prediction.
            """
        )

        try:

            with st.spinner(
                "Generating Grad-CAM..."
            ):

                (
                    backbone,
                    last_conv_layer,
                    grad_model,
                    head_layers
                ) = create_gradcam_components(
                    model
                )

                predicted_index = int(
                    np.argmax(probabilities)
                )

                heatmap = generate_gradcam(
                    image,
                    grad_model,
                    head_layers,
                    predicted_index
                )

                overlay = create_heatmap_overlay(
                    image,
                    heatmap
                )

            grad_col1, grad_col2 = st.columns(
                2
            )

            with grad_col1:

                st.image(
                    image,
                    caption="Original MRI",
                    use_container_width=True
                )

            with grad_col2:

                st.image(
                    overlay,
                    caption="Grad-CAM",
                    use_container_width=True
                )

        except Exception as e:

            st.warning(
                "Prediction was successful, but "
                "Grad-CAM could not be generated."
            )

            st.exception(e)

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.divider()

    st.caption(
        "Brain Tumor Detection and Classification Using "
        "Deep Learning on MRI Images"
    )

    st.caption(
        "Final model: ResNet50 | Test Accuracy: 93.31%"
    )

    st.caption(
        "Educational/research prototype — "
        "not a medical diagnostic tool."
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    main()