
# -*- coding: utf-8 -*-
"""
Soil Particle Size Distribution (PSD) Analyzer
Predicts soil PSD curves from close-range smartphone images
using a Convolutional Neural Network.
"""

import os
import io
import math
import json

import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image as PILImage, ImageOps
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import plotly.graph_objects as go
import tensorflow as tf


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "CNN.h5")

SAMPLE_PATH = os.path.join(
    BASE_DIR, "sample_images", "soil_sample.jpg"
)


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------

st.set_page_config(
    page_title="Soil Particle Size Distribution Analyzer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.metric-container {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 12px;
}

.metric-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #1e293b;
    margin: 4px 0;
}

.metric-label {
    font-size: 0.85rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

.metric-sub {
    font-size: 0.8rem;
    color: #94a3b8;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #0f172a;
    margin-top: 1rem;
    margin-bottom: 0.75rem;
}

.badge-pill {
    display: inline-block;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: 9999px;
    background-color: #e0f2fe;
    color: #0369a1;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# TensorFlow / Keras Compatibility
# ---------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class CompatibleDepthwiseConv2D(
    tf.keras.layers.DepthwiseConv2D
):
    """
    Compatibility layer for H5 models containing
    DepthwiseConv2D configurations with groups=1.

    The groups argument is not accepted by some
    TensorFlow/Keras versions.
    """

    @classmethod
    def from_config(cls, config):

        config = dict(config)

        # groups=1 is the default depthwise behavior.
        # Remove it for older Keras implementations.
        config.pop("groups", None)

        return cls(**config)


# ---------------------------------------------------------
# Model Loading (Cached)
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading CNN model...")

def load_model(model_path=MODEL_PATH):

    model_path = os.path.abspath(model_path)

    # Check whether the file exists.
    if not os.path.isfile(model_path):
        return None, (
            f"Model file does not exist: {model_path}"
        )

    try:

        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "DepthwiseConv2D":
                    CompatibleDepthwiseConv2D,

                "CompatibleDepthwiseConv2D":
                    CompatibleDepthwiseConv2D
            },
            compile=False
        )

        return model, None

    except Exception as e:

        error_message = (
            f"{type(e).__name__}: {str(e)}"
        )

        return None, error_message


# ---------------------------------------------------------
# Prediction Output Validation
# ---------------------------------------------------------

def extract_scalar(output, parameter_name):

    arr = np.asarray(output)

    if arr.size == 0:
        raise ValueError(
            f"Empty prediction for parameter {parameter_name}"
        )

    value = float(arr.reshape(-1)[0])

    if not np.isfinite(value):
        raise ValueError(
            f"Invalid prediction for {parameter_name}: {value}"
        )

    return value


def predict_parameters(model, input_tensor):

    predictions = model.predict(
        input_tensor,
        verbose=0
    )

    if isinstance(predictions, dict):

        if len(predictions) != 2:
            raise ValueError(
                "Expected exactly two model outputs (b and c)."
            )

        values = list(predictions.values())

    elif isinstance(predictions, (list, tuple)):

        if len(predictions) != 2:
            raise ValueError(
                f"Expected 2 model outputs, got {len(predictions)}."
            )

        values = predictions

    else:

        raise ValueError(
            "Model must return two outputs: b and c."
        )

    raw_b = extract_scalar(values[0], "b")
    raw_c = extract_scalar(values[1], "c")

    # The original model predicts log(b).
    b_val = math.exp(raw_b)

    # The original model predicts c directly.
    c_val = raw_c

    if not np.isfinite(b_val) or b_val <= 0:
        raise ValueError(
            f"Invalid b parameter: {b_val}"
        )

    if not np.isfinite(c_val) or c_val <= 0:
        raise ValueError(
            f"Invalid c parameter: {c_val}"
        )

    return b_val, c_val


# ---------------------------------------------------------
# Image Preprocessing & Crop Functions
# ---------------------------------------------------------

DEVICE_PROFILES = {
    "Auto-Detect (from resolution)": None,

    "Motorola Edge (1800 x 4000)": 720 / 1800,

    "Samsung Galaxy A52 (6936 x 9248)": 1600 / 9248,

    "Apple iPhone 14 (3024 x 4032)": 552 / 1673,

    "Custom Scaling Factor": "custom"
}


def detect_device_scale(width, height):

    dims = {width, height}

    if 1800 in dims or 4000 in dims:
        return 720 / 1800, (
            "Motorola Edge (1800x4000 detected)"
        )

    if 6936 in dims or 9248 in dims:
        return 1600 / 9248, (
            "Samsung Galaxy A52 (6936x9248 detected)"
        )

    if 3024 in dims or 4032 in dims:
        return 552 / 1673, (
            "iPhone 14 / HPC (3024x4032 detected)"
        )

    return 1.0, "Standard resolution (Scale = 1.0)"


def preprocess_and_crop(image, scale, crop_size=400):

    orig_w, orig_h = image.size

    new_w = max(int(orig_w * scale), crop_size)
    new_h = max(int(orig_h * scale), crop_size)

    resized_image = image.resize(
        (new_w, new_h),
        PILImage.Resampling.LANCZOS
    )

    left = (new_w - crop_size) // 2
    upper = (new_h - crop_size) // 2

    cropped_image = resized_image.crop((
        left,
        upper,
        left + crop_size,
        upper + crop_size
    ))

    img_rgb = cropped_image.convert("RGB")

    img_array = np.array(
        img_rgb,
        dtype=np.float32
    )

    img_batch = np.expand_dims(
        img_array,
        axis=0
    )

    processed_tensor = (
        tf.keras.applications.mobilenet.preprocess_input(
            img_batch
        )
    )

    return (
        cropped_image,
        processed_tensor,
        (orig_w, orig_h),
        (new_w, new_h)
    )


# ---------------------------------------------------------
# Geotechnical & PSD Computations
# ---------------------------------------------------------

def rosin_rammler_curve(d_arr, b, c):

    return 1.0 - np.exp(
        -((d_arr / b) ** c)
    )


def compute_diameter(p, b, c):

    val = -math.log(1.0 - p)

    return b * (val ** (1.0 / c))


def compute_soil_fractions(b, c):

    y_clay = 1.0 - math.exp(
        -((0.002 / b) ** c)
    )

    y_silt_top = 1.0 - math.exp(
        -((0.063 / b) ** c)
    )

    y_sand_top = 1.0 - math.exp(
        -((2.0 / b) ** c)
    )

    y_gravel_top = 1.0 - math.exp(
        -((63.0 / b) ** c)
    )

    f_clay = max(0.0, y_clay) * 100

    f_silt = max(
        0.0, y_silt_top - y_clay
    ) * 100

    f_sand = max(
        0.0, y_sand_top - y_silt_top
    ) * 100

    f_gravel = max(
        0.0, y_gravel_top - y_sand_top
    ) * 100

    f_boulders = max(
        0.0, 1.0 - y_gravel_top
    ) * 100

    return {
        "Clay (< 0.002 mm)": f_clay,
        "Silt (0.002 - 0.063 mm)": f_silt,
        "Sand (0.063 - 2.0 mm)": f_sand,
        "Gravel (2.0 - 63.0 mm)": f_gravel,
        "Cobbles / Boulders (> 63 mm)": f_boulders
    }


# ---------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------

def create_plotly_psd(b, c, d_vals, y_vals, d10, d30, d50, d60):
    """Interactive PSD plot with logarithmic particle-size axis."""
    fig = go.Figure()

    zones = [
        ("Clay (Cl)", 0.001, 0.002, "rgba(219,234,254,0.55)"),
        ("Silt (Si)", 0.002, 0.063, "rgba(220,252,231,0.55)"),
        ("Sand (Sa)", 0.063, 2.0, "rgba(254,243,199,0.55)"),
        ("Gravel (Gr)", 2.0, 63.0, "rgba(254,226,226,0.55)"),
        ("Cobbles (Cb)", 63.0, 200.0, "rgba(243,244,246,0.65)")
    ]

    for name, x0, x1, color in zones:
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=color,
            layer="below",
            line_width=0
        )
        fig.add_annotation(
            x=(x0 * x1) ** 0.5,
            y=103,
            xref="x",
            yref="y",
            text=f"<b>{name}</b>",
            showarrow=False,
            font=dict(size=10, color="#334155")
        )

    # Predicted percentage-finer curve
    fig.add_trace(go.Scatter(
        x=d_vals,
        y=np.asarray(y_vals) * 100,
        mode="lines",
        name="Predicted PSD Curve",
        line=dict(color="#0284c7", width=3),
        hovertemplate=(
            "<b>Particle size:</b> %{x:.4g} mm"
            "<br><b>Percent finer:</b> %{y:.2f}%"
            "<extra></extra>"
        )
    ))

    # Characteristic diameter markers (labels appear on hover to avoid overlap)
    points = [
        ("D10", d10, 10),
        ("D30", d30, 30),
        ("D50", d50, 50),
        ("D60", d60, 60)
    ]
    valid_points = [
        p for p in points
        if np.isfinite(p[1]) and 0.001 <= p[1] <= 200
    ]

    if valid_points:
        fig.add_trace(go.Scatter(
            x=[p[1] for p in valid_points],
            y=[p[2] for p in valid_points],
            mode="markers",
            name="Characteristic Diameters",
            marker=dict(
                size=10,
                color="#dc2626",
                symbol="diamond",
                line=dict(width=1.5, color="#ffffff")
            ),
            customdata=[[p[0], p[1]] for p in valid_points],
            hovertemplate=(
                "<b>%{customdata[0]}</b>: %{customdata[1]:.4g} mm"
                "<br>Percent finer: %{y:.0f}%"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title=dict(
            text="Particle Size Distribution (PSD) Curve",
            x=0.02
        ),
        xaxis=dict(
            type="log",
            title="Particle size, d (mm)",
            range=[np.log10(0.001), np.log10(200)],
            tickmode="array",
            tickvals=[
                0.001, 0.002, 0.0063, 0.02,
                0.063, 0.2, 0.63, 2,
                6.3, 20, 63, 200
            ],
            ticktext=[
                "0.001", "0.002", "0.0063", "0.02",
                "0.063", "0.2", "0.63", "2",
                "6.3", "20", "63", "200"
            ],
            showgrid=True,
            gridcolor="#e2e8f0",
            minor=dict(showgrid=True, gridcolor="#f1f5f9")
        ),
        yaxis=dict(
            title="Percent finer (%)",
            range=[0, 107],
            tickmode="linear",
            dtick=10,
            ticksuffix="%",
            showgrid=True,
            gridcolor="#e2e8f0"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="right",
            x=1
        ),
        margin=dict(l=65, r=30, t=95, b=60),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="closest",
        height=520
    )

    return fig


def create_matplotlib_psd(b, c, d_vals, y_vals, d10, d30, d50, d60):
    """Publication-quality PSD plot with characteristic diameter guides."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)

    zones = [
        (0.001, 0.002, "#dbeafe", "Cl"),
        (0.002, 0.063, "#dcfce7", "Si"),
        (0.063, 2.0, "#fef3c7", "Sa"),
        (2.0, 63.0, "#fee2e2", "Gr"),
        (63.0, 200.0, "#f3f4f6", "Cb")
    ]

    for left, right, color, label in zones:
        ax.axvspan(left, right, color=color, alpha=0.65, zorder=0)
        ax.text(
            np.sqrt(left * right),
            1.025,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
            clip_on=False
        )

    ax.plot(
        d_vals,
        y_vals,
        color="#0284c7",
        linewidth=2.5,
        label="Predicted PSD Curve",
        zorder=3
    )

    characteristic = [
        ("D10", d10, 0.10),
        ("D30", d30, 0.30),
        ("D50", d50, 0.50),
        ("D60", d60, 0.60)
    ]

    for name, diameter, probability in characteristic:
        if np.isfinite(diameter) and 0.001 <= diameter <= 200:
            ax.scatter(
                diameter,
                probability,
                color="#dc2626",
                edgecolor="white",
                linewidth=0.8,
                s=45,
                zorder=5
            )
            ax.axhline(
                probability,
                color="#94a3b8",
                linestyle=":",
                linewidth=0.7,
                zorder=1
            )
            ax.axvline(
                diameter,
                color="#94a3b8",
                linestyle=":",
                linewidth=0.7,
                zorder=1
            )
            ax.annotate(
                f"{name} = {diameter:.3g} mm",
                xy=(diameter, probability),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=7.5,
                color="#991b1b"
            )

    ax.set_xscale("log")
    ax.set_xlim(0.001, 200)
    ax.set_ylim(0, 1.08)

    ticks = [
        0.001, 0.002, 0.0063, 0.02,
        0.063, 0.2, 0.63, 2,
        6.3, 20, 63, 200
    ]
    ax.set_xticks(ticks)
    ax.set_xticklabels([
        "0.001", "0.002", "0.0063", "0.02",
        "0.063", "0.2", "0.63", "2",
        "6.3", "20", "63", "200"
    ])

    ax.yaxis.set_major_locator(mtick.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(
        mtick.PercentFormatter(xmax=1.0, decimals=0)
    )

    ax.set_xlabel("Particle size, d (mm)", fontsize=10)
    ax.set_ylabel("Percent finer (%)", fontsize=10)
    ax.set_title(
        "Particle Size Distribution (PSD) Curve",
        fontsize=13,
        pad=22
    )

    ax.grid(which="major", linewidth=0.6, color="#cbd5e1")
    ax.grid(
        which="minor",
        linewidth=0.35,
        color="#e2e8f0",
        linestyle=":"
    )
    ax.set_axisbelow(True)

    ax.text(
        0.02,
        0.96,
        rf"$F(d)=1-\exp[-(d/b)^c]$" +
        f"\nb = {b:.4g} mm\nc = {c:.4g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#cbd5e1",
            alpha=0.9
        )
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------

model, model_error = load_model(MODEL_PATH)

with st.sidebar:

    st.header("⚙️ Configuration")

    if model is not None:

        st.success(
            "Deep Learning Model (CNN.h5) Loaded",
            icon="🧠"
        )

        st.caption(
            f"TensorFlow version: {tf.__version__}"
        )

    else:

        if not os.path.isfile(MODEL_PATH):

            st.error(
                "CNN.h5 was not found in the application directory."
            )

            st.caption(
                f"Expected location: {MODEL_PATH}"
            )

        else:

            st.error(
                "CNN.h5 exists, but model loading failed."
            )

            st.code(
                model_error or "Unknown model loading error."
            )

    st.markdown("---")

    st.subheader("📱 Device & Scaling")

    selected_device = st.selectbox(
        "Smartphone Model / Profile",
        options=list(DEVICE_PROFILES.keys()),
        index=0,
        help=(
            "Select the smartphone camera used to "
            "photograph the soil sample."
        )
    )

    scale_override = None

    if selected_device == "Custom Scaling Factor":

        scale_override = st.number_input(
            "Custom Scale Factor",
            min_value=0.01,
            max_value=2.0,
            value=0.40,
            step=0.01
        )

    elif selected_device != "Auto-Detect (from resolution)":

        scale_override = DEVICE_PROFILES[selected_device]

    st.markdown("---")

    st.subheader("🧪 Quick Test")

    if os.path.isfile(SAMPLE_PATH):

        if st.button(
            "📸 Load Sample Soil Image",
            width="stretch"
        ):

            st.session_state["use_sample_clicked"] = True

    else:

        st.caption(
            "No sample image found in sample_images/soil_sample.jpg"
        )

    if st.button(
        "Clear Sample Selection",
        width="stretch"
    ):

        st.session_state["use_sample_clicked"] = False


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------

st.title(
    "🌱 Soil Particle Size Distribution (PSD) Analyzer"
)

st.markdown("""
Upload a top-down close-range smartphone photo of a soil sample
to predict its **particle size distribution curve** and
**geotechnical gradation parameters**
($b, c, D_{10}, D_{30}, D_{50}, D_{60}, C_u, C_c$).
""")

uploaded_file = st.file_uploader(
    "Choose a soil photograph (JPG, JPEG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    help=(
        "For optimal accuracy, use a close-range, "
        "well-lit top-down photo."
    )
)


# ---------------------------------------------------------
# Image Selection
# ---------------------------------------------------------

image_to_process = None
image_source_label = ""

if uploaded_file is not None:

    try:

        image_to_process = PILImage.open(
            uploaded_file
        )

        image_to_process = ImageOps.exif_transpose(
            image_to_process
        )

        image_source_label = (
            f"Uploaded: {uploaded_file.name}"
        )

    except Exception as e:

        st.error(
            f"Could not open uploaded image: {e}"
        )

elif (
    st.session_state.get("use_sample_clicked", False)
    and os.path.isfile(SAMPLE_PATH)
):

    try:

        image_to_process = PILImage.open(
            SAMPLE_PATH
        ).convert("RGB")

        image_source_label = (
            "Built-in Sample Soil Photo"
        )

    except Exception as e:

        st.error(
            f"Could not load sample image: {e}"
        )


# ---------------------------------------------------------
# Processing & Prediction Pipeline
# ---------------------------------------------------------

if image_to_process is not None:

    if model is None:

        st.error(
            "Prediction unavailable because CNN.h5 "
            "could not be loaded."
        )

        st.info(
            "Check the model-loading error in the sidebar."
        )

    else:

        orig_w, orig_h = image_to_process.size

        if scale_override is not None:

            active_scale = scale_override

            device_desc = (
                f"{selected_device} "
                f"(scale: {active_scale:.4f})"
            )

        else:

            active_scale, device_desc = (
                detect_device_scale(orig_w, orig_h)
            )

        try:

            with st.spinner(
                "Processing image and executing neural network inference..."
            ):

                (
                    cropped_pil,
                    input_tensor,
                    orig_dims,
                    resized_dims
                ) = preprocess_and_crop(
                    image_to_process,
                    scale=active_scale,
                    crop_size=400
                )

                # CNN inference
                b_val, c_val = predict_parameters(
                    model,
                    input_tensor
                )

                # Characteristic diameters
                d10 = compute_diameter(
                    0.10, b_val, c_val
                )

                d30 = compute_diameter(
                    0.30, b_val, c_val
                )

                d50 = compute_diameter(
                    0.50, b_val, c_val
                )

                d60 = compute_diameter(
                    0.60, b_val, c_val
                )

                # Gradation coefficients
                cu = (
                    d60 / d10
                    if d10 > 0 else 0.0
                )

                cc = (
                    (d30 ** 2) / (d10 * d60)
                    if (d10 * d60) > 0 else 0.0
                )

                # Soil fractions
                fractions = compute_soil_fractions(
                    b_val, c_val
                )

                # PSD curve
                d_points = np.logspace(
                    -3, 2.3, num=200
                )

                y_points = rosin_rammler_curve(
                    d_points,
                    b_val,
                    c_val
                )

            # -------------------------------------------------
            # Image Inspection
            # -------------------------------------------------

            st.markdown(
                '<div class="section-title">'
                '📷 Image Inspection & Crop ROI</div>',
                unsafe_allow_html=True
            )

            col_img1, col_img2 = st.columns(
                [1.2, 1]
            )

            with col_img1:

                st.image(
                    image_to_process,
                    caption=(
                        f"Original Photograph "
                        f"({orig_w} × {orig_h} px)"
                    ),
                    width="stretch"
                )

                st.caption(
                    f"**Source:** {image_source_label} "
                    f"| **Profile:** {device_desc}"
                )

            with col_img2:

                st.image(
                    cropped_pil,
                    caption=(
                        "Rescaled & Center-Cropped Input "
                        "to CNN (400 × 400 px)"
                    ),
                    width="stretch"
                )

                st.info(
                    f"**Active Scale Factor:** "
                    f"`{active_scale:.4f}` | "
                    f"Resized to `{resized_dims[0]}×"
                    f"{resized_dims[1]}` px."
                )

            st.markdown("---")

            # -------------------------------------------------
            # Predicted Parameters
            # -------------------------------------------------

            st.markdown(
                '<div class="section-title">'
                '📊 Predicted Rosin-Rammler & '
                'Geotechnical Parameters</div>',
                unsafe_allow_html=True
            )

            metric_values = [
                ("Scale (b)", f"{b_val:.3f} mm",
                 "Rosin-Rammler b"),

                ("Gradation (c)", f"{c_val:.3f}",
                 "Uniformity exponent"),

                ("D₁₀ (Effective)", f"{d10:.3f} mm",
                 "10% Passing"),

                ("D₅₀ (Median)", f"{d50:.3f} mm",
                 "50% Passing"),

                ("Cu (Uniformity)", f"{cu:.2f}",
                 "D₆₀ / D₁₀"),

                ("Cc (Curvature)", f"{cc:.2f}",
                 "(D₃₀)² / (D₁₀·D₆₀)")
            ]

            metric_cols = st.columns(6)

            for col, (label, value, sub) in zip(
                metric_cols, metric_values
            ):

                with col:

                    st.markdown(
                        f"""
                        <div class="metric-container">
                            <div class="metric-label">
                                {label}
                            </div>
                            <div class="metric-value">
                                {value}
                            </div>
                            <div class="metric-sub">
                                {sub}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown("---")

            # -------------------------------------------------
            # PSD Curves
            # -------------------------------------------------

            st.markdown(
                '<div class="section-title">'
                '📈 Particle Size Distribution Curves'
                '</div>',
                unsafe_allow_html=True
            )

            _, tab_publication, tab_fractions = (
                st.tabs([
                    "✨ Interactive Plotly Curve",
                    "📄 Scientific Matplotlib Plot",
                    "🧩 Soil Fractions Breakdown"
                ])
            )

            with tab_interactive:

                plotly_fig = create_plotly_psd(
                    b_val,
                    c_val,
                    d_points,
                    y_points,
                    d10,
                    d30,
                    d50,
                    d60
                )

                st.plotly_chart(
                    plotly_fig,
                    width="stretch"
                )

            with tab_publication:

                mpl_fig = create_matplotlib_psd(
                    b_val,
                    c_val,
                    d_points,
                    y_points,
                    d10,
                    d30,
                    d50,
                    d60
                )

                st.pyplot(mpl_fig)

                buf = io.BytesIO()

                mpl_fig.savefig(
                    buf,
                    format="png",
                    dpi=300,
                    bbox_inches="tight"
                )

                st.download_button(
                    label=(
                        "📥 Download Publication Figure "
                        "(PNG, 300 DPI)"
                    ),
                    data=buf.getvalue(),
                    file_name="soil_psd_curve.png",
                    mime="image/png"
                )

                plt.close(mpl_fig)

            with tab_fractions:

                st.write(
                    "#### Geotechnical Grain-Size "
                    "Classification Breakdown"
                )

                f_col1, f_col2 = st.columns(2)

                with f_col1:

                    fraction_df = pd.DataFrame([
                        {
                            "Fraction": k,
                            "Percentage (%)": round(v, 2)
                        }
                        for k, v in fractions.items()
                    ])

                    st.dataframe(
                        fraction_df,
                        hide_index=True,
                        width="stretch"
                    )

                with f_col2:

                    for frac_name, frac_val in (
                        fractions.items()
                    ):

                        st.write(
                            f"**{frac_name}**: "
                            f"`{frac_val:.1f}%`"
                        )

                        st.progress(
                            min(
                                max(
                                    float(frac_val) / 100.0,
                                    0.0
                                ),
                                1.0
                            )
                        )

            st.markdown("---")

            # -------------------------------------------------
            # Data Export
            # -------------------------------------------------

            st.markdown(
                '<div class="section-title">'
                '💾 Export Data & Report</div>',
                unsafe_allow_html=True
            )

            exp_col1, exp_col2 = st.columns(2)

            with exp_col1:

                csv_df = pd.DataFrame({
                    "particle_size_d_mm":
                        np.round(d_points, 5),

                    "fraction_finer_y":
                        np.round(y_points, 5),

                    "percent_finer":
                        np.round(y_points * 100, 2)
                })

                csv_data = csv_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label=(
                        "📥 Download Curve Data Points (CSV)"
                    ),
                    data=csv_data,
                    file_name="predicted_psd_curve.csv",
                    mime="text/csv",
                    width="stretch"
                )

            with exp_col2:

                report = {
                    "source": image_source_label,

                    "profile": device_desc,

                    "scale_factor": active_scale,

                    "parameters": {
                        "rosin_rammler_b_mm":
                            round(b_val, 4),

                        "rosin_rammler_c":
                            round(c_val, 4)
                    },

                    "characteristic_diameters_mm": {
                        "D10": round(d10, 4),
                        "D30": round(d30, 4),
                        "D50": round(d50, 4),
                        "D60": round(d60, 4)
                    },

                    "coefficients": {
                        "Cu": round(cu, 2),
                        "Cc": round(cc, 2)
                    },

                    "soil_fractions_percent": {
                        k: round(v, 2)
                        for k, v in fractions.items()
                    }
                }

                json_str = json.dumps(
                    report,
                    indent=2
                )

                st.download_button(
                    label=(
                        "📥 Download Full Analysis Report (JSON)"
                    ),
                    data=json_str,
                    file_name="soil_psd_analysis_report.json",
                    mime="application/json",
                    width="stretch"
                )

        except Exception as e:

            st.error(
                "An error occurred during image processing "
                "or model inference."
            )

            st.exception(e)

else:

    st.info(
        "👆 Please upload a soil image from your computer "
        "or click **'Load Sample Soil Image'** in the sidebar "
        "to test immediately."
    )
