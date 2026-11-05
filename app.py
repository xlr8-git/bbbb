# -*- coding: utf-8 -*-
"""
Soil Particle Size Distribution (PSD) Analyzer
Predicts soil PSD curves from close-range smartphone images using a Convolutional Neural Network.
"""

import os
import io
import math
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import plotly.graph_objects as go
import tensorflow as tf

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Soil Particle Size Distribution Analyzer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern UI
st.markdown("""
<style>
    /* Metric Card Styling */
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
    /* Section Headers */
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0f172a;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    /* Info badge */
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #e0f2fe;
        color: #0369a1;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Model Loading (Cached)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading deep learning model (CNN.h5)...")
def load_model(model_path="CNN.h5"):
    """Loads the pre-trained Keras model once and caches it."""
    if not os.path.exists(model_path):
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None


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
    """Detects appropriate scale factor based on known smartphone camera resolutions."""
    # Check both orientations
    dims = {width, height}
    if 1800 in dims or 4000 in dims:
        return 720 / 1800, "Motorola Edge (1800x4000 detected)"
    if 6936 in dims or 9248 in dims:
        return 1600 / 9248, "Samsung Galaxy A52 (6936x9248 detected)"
    if 3024 in dims or 4032 in dims:
        return 552 / 1673, "iPhone 14 / HPC (3024x4032 detected)"
    return 1.0, "Standard resolution (Scale = 1.0)"

def preprocess_and_crop(image, scale, crop_size=400):
    """Rescales and center-crops the input PIL image to (crop_size, crop_size)."""
    orig_w, orig_h = image.size
    new_w = max(int(orig_w * scale), crop_size)
    new_h = max(int(orig_h * scale), crop_size)

    # High-quality resize
    resized_image = image.resize((new_w, new_h), PILImage.Resampling.LANCZOS)

    # Center crop
    left = (new_w - crop_size) // 2
    upper = (new_h - crop_size) // 2
    right = left + crop_size
    lower = upper + crop_size
    cropped_image = resized_image.crop((left, upper, right, lower))

    # Convert to RGB float array for MobileNet
    img_rgb = cropped_image.convert("RGB")
    img_array = np.array(img_rgb, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0) # shape: (1, 400, 400, 3)

    # MobileNet preprocessing
    processed_tensor = tf.keras.applications.mobilenet.preprocess_input(img_batch)
    return cropped_image, processed_tensor, (orig_w, orig_h), (new_w, new_h)


# ---------------------------------------------------------
# Geotechnical & PSD Computations
# ---------------------------------------------------------
def rosin_rammler_curve(d_arr, b, c):
    """Computes y = 1 - exp(-(d/b)^c) where d is particle diameter in mm."""
    return 1.0 - np.exp(-((d_arr / b) ** c))

def compute_diameter(p, b, c):
    """Computes particle diameter d (mm) for a given cumulative passing fraction p (0 < p < 1)."""
    # From y = 1 - exp(-(d/b)^c) => d = b * (-ln(1 - p))^(1/c)
    val = -math.log(1.0 - p)
    return b * (val ** (1.0 / c))

def compute_soil_fractions(b, c):
    """Calculates geotechnical soil fractions according to standard particle size boundaries."""
    # Boundaries (mm):
    # Clay: < 0.002 mm
    # Silt: 0.002 to 0.063 mm
    # Sand: 0.063 to 2.0 mm
    # Gravel: 2.0 to 63.0 mm
    # Cobbles/Boulders: >= 63.0 mm
    y_clay = 1.0 - math.exp(-((0.002 / b) ** c))
    y_silt_top = 1.0 - math.exp(-((0.063 / b) ** c))
    y_sand_top = 1.0 - math.exp(-((2.0 / b) ** c))
    y_gravel_top = 1.0 - math.exp(-((63.0 / b) ** c))

    f_clay = max(0.0, y_clay) * 100
    f_silt = max(0.0, y_silt_top - y_clay) * 100
    f_sand = max(0.0, y_sand_top - y_silt_top) * 100
    f_gravel = max(0.0, y_gravel_top - y_sand_top) * 100
    f_boulders = max(0.0, 1.0 - y_gravel_top) * 100

    return {
        "Clay (< 0.002 mm)": f_clay,
        "Silt (0.002 - 0.063 mm)": f_silt,
        "Sand (0.063 - 2.0 mm)": f_sand,
        "Gravel (2.0 - 63.0 mm)": f_gravel,
        "Cobbles / Boulders (> 63 mm)": f_boulders
    }


# ---------------------------------------------------------
# Plotting Functions (Plotly & Matplotlib)
# ---------------------------------------------------------
def create_plotly_psd(b, c, d_vals, y_vals, d10, d30, d50, d60):
    """Generates an interactive Plotly PSD curve with shaded soil zones."""
    fig = go.Figure()

    # Soil classification zones (Clay, Silt, Sand, Gravel, Cobbles)
    zones = [
        {"name": "Clay (Cl)", "x0": 0.001, "x1": 0.002, "x_label": 0.0014, "color": "rgba(219, 234, 254, 0.45)"},
        {"name": "Silt (Si)", "x0": 0.002, "x1": 0.063, "x_label": 0.011, "color": "rgba(220, 252, 231, 0.45)"},
        {"name": "Sand (Sa)", "x0": 0.063, "x1": 2.0, "x_label": 0.35, "color": "rgba(254, 243, 199, 0.45)"},
        {"name": "Gravel (Gr)", "x0": 2.0, "x1": 63.0, "x_label": 11.0, "color": "rgba(254, 226, 226, 0.45)"},
        {"name": "Cobbles (Bo)", "x0": 63.0, "x1": 200.0, "x_label": 112.0, "color": "rgba(243, 244, 246, 0.55)"}
    ]

    for z in zones:
        # Shaded background
        fig.add_vrect(
            x0=z["x0"], x1=z["x1"],
            fillcolor=z["color"], opacity=1.0,
            layer="below", line_width=1, line_color="rgba(148, 163, 184, 0.4)"
        )
        # Zone header label
        fig.add_annotation(
            x=math.log10(z["x_label"]),
            y=102,
            xref="x", yref="y",
            text=f"<b>{z['name']}</b>",
            showarrow=False,
            font=dict(size=11, color="#334155")
        )

    # PSD Curve
    fig.add_trace(go.Scatter(
        x=d_vals,
        y=y_vals * 100,
        mode="lines",
        name="Predicted PSD Curve",
        line=dict(color="#0284c7", width=3.5),
        hovertemplate="<b>Particle size d:</b> %{x:.4f} mm<br><b>Percent finer:</b> %{y:.1f}%<extra></extra>"
    ))

    # Characteristic diameters
    pts = [
        ("D10", d10, 10),
        ("D30", d30, 30),
        ("D50", d50, 50),
        ("D60", d60, 60)
    ]
    pt_x = [p[1] for p in pts if 0.001 <= p[1] <= 200]
    pt_y = [p[2] for p in pts if 0.001 <= p[1] <= 200]
    pt_text = [f"{p[0]} = {p[1]:.3f} mm" for p in pts if 0.001 <= p[1] <= 200]

    if pt_x:
        fig.add_trace(go.Scatter(
            x=pt_x,
            y=pt_y,
            mode="markers+text",
            name="Key Diameters (D10, D30, D50, D60)",
            marker=dict(size=10, color="#dc2626", symbol="diamond", line=dict(width=1.5, color="#ffffff")),
            text=pt_text,
            textposition="top left",
            textfont=dict(size=11, color="#991b1b"),
            hovertemplate="<b>%{text}</b><br>Percent passing: %{y}%<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text="Particle Size Distribution (PSD) Curve",
            font=dict(size=16, color="#0f172a", family="sans-serif")
        ),
        xaxis=dict(
            type="log",
            title="Particle size d (mm)",
            range=[math.log10(0.001), math.log10(200)],
            tickvals=[0.001, 0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200],
            ticktext=["0.001", "0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"],
            gridcolor="#e2e8f0",
            showgrid=True
        ),
        yaxis=dict(
            title="Percent Finer (%)",
            range=[0, 107],
            ticksuffix="%",
            gridcolor="#e2e8f0",
            showgrid=True
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        ),
        margin=dict(l=45, r=30, t=65, b=45),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=480
    )
    return fig

def create_matplotlib_psd(b, c, d_vals, y_vals):
    """Renders the publication-format Matplotlib chart matching the research paper."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=220)
    ax.grid(which='major', linewidth=0.6, color='#cbd5e1')
    ax.grid(which='minor', linewidth=0.25, color='#e2e8f0', linestyle=':')
    ax.semilogx()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    
    ticks = [0.001, 0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=7)
    ax.set_xlim([0.001, 200])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('Particle size $d$ (mm)', fontsize=9)
    ax.set_ylabel('Finer', fontsize=9)

    # Soil division lines
    for boundary in [0.002, 0.063, 2.0, 63.0]:
        ax.plot([boundary, boundary], [0, 1], c='#334155', linewidth=0.8, linestyle='--')

    # Boundary labels
    ax.text(0.0013, 1.01, 'Cl', fontsize=8, fontweight='bold', color='#1e293b')
    ax.text(0.011, 1.01, 'Si', fontsize=8, fontweight='bold', color='#1e293b')
    ax.text(0.35, 1.01, 'Sa', fontsize=8, fontweight='bold', color='#1e293b')
    ax.text(10, 1.01, 'Gr', fontsize=8, fontweight='bold', color='#1e293b')
    ax.text(100, 1.01, 'Bo', fontsize=8, fontweight='bold', color='#1e293b')

    # Formula & parameter annotations
    ax.text(0.0012, 0.93, r'$y = 1 - \exp(-(d/b)^c)$', fontsize=8, color='#0f172a')
    ax.text(0.0012, 0.85, f'b = {b:.3f} mm', fontsize=8, color='#0369a1')
    ax.text(0.0012, 0.77, f'c = {c:.3f}', fontsize=8, color='#0369a1')

    # Fitted line
    ax.plot(d_vals, y_vals, linewidth=1.8, c='#0284c7', label='Predicted PSD')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# Sidebar Controls & Information
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # Model status
    model = load_model("CNN.h5")
    if model is not None:
        st.success("✅ Deep Learning Model (`CNN.h5`) Loaded", icon="🧠")
    else:
        st.error("❌ Model file `CNN.h5` not found in directory.")

    st.markdown("---")
    st.subheader("📱 Device & Scaling")
    selected_device = st.selectbox(
        "Smartphone Model / Profile",
        options=list(DEVICE_PROFILES.keys()),
        index=0,
        help="Select the smartphone camera used to photograph the soil sample. The model uses device-calibrated scaling."
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
    sample_path = os.path.join("sample_images", "soil_sample.jpg")
    use_sample = False
    if os.path.exists(sample_path):
        if st.button("📸 Load Sample Soil Image", use_container_width=True):
            st.session_state["use_sample_clicked"] = True



# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.title("🌱 Soil Particle Size Distribution (PSD) Analyzer")
st.markdown("""
Upload a top-down close-range smartphone photo of a soil sample to predict its **particle size distribution curve** 
and **geotechnical gradation parameters** ($b, c, D_{10}, D_{30}, D_{50}, D_{60}, C_u, C_c$).
""")

# File Uploader
uploaded_file = st.file_uploader(
    "Choose a soil photograph (JPG, JPEG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    help="For optimal accuracy, use a close-range, well-lit top-down photo."
)

# Handle sample image trigger
image_to_process = None
image_source_label = ""

if uploaded_file is not None:
    try:
        image_to_process = PILImage.open(uploaded_file)
        image_source_label = f"Uploaded: {uploaded_file.name}"
    except Exception as e:
        st.error(f"Could not open uploaded image: {e}")
elif st.session_state.get("use_sample_clicked", False) and os.path.exists(sample_path):
    image_to_process = PILImage.open(sample_path)
    image_source_label = "Built-in Sample Soil Photo"


# ---------------------------------------------------------
# Processing & Prediction Pipeline
# ---------------------------------------------------------
if image_to_process is not None:
    if model is None:
        st.error("Cannot proceed: Pre-trained CNN model `CNN.h5` is not loaded.")
    else:
        orig_w, orig_h = image_to_process.size

        # Determine scaling factor
        if scale_override is not None:
            active_scale = scale_override
            device_desc = f"{selected_device} (scale: {active_scale:.4f})"
        else:
            active_scale, device_desc = detect_device_scale(orig_w, orig_h)

        # Preprocess and crop
        with st.spinner("Processing image and executing neural network inference..."):
            cropped_pil, input_tensor, orig_dims, resized_dims = preprocess_and_crop(
                image_to_process,
                scale=active_scale,
                crop_size=400
            )

            # Model inference
            preds = model.predict(input_tensor, verbose=0)
            # Parameter b
            raw_b = float(preds[0][0][0]) if len(preds[0].shape) > 1 else float(preds[0][0])
            b_val = math.exp(raw_b)

            # Parameter c
            raw_c = float(preds[1][0][0]) if len(preds[1].shape) > 1 else float(preds[1][0])
            c_val = raw_c

            # Characteristic diameters
            d10 = compute_diameter(0.10, b_val, c_val)
            d30 = compute_diameter(0.30, b_val, c_val)
            d50 = compute_diameter(0.50, b_val, c_val)
            d60 = compute_diameter(0.60, b_val, c_val)

            # Gradation coefficients
            cu = (d60 / d10) if d10 > 0 else 0.0
            cc = ((d30 ** 2) / (d10 * d60)) if (d10 * d60) > 0 else 0.0

            # Soil fraction breakdown
            fractions = compute_soil_fractions(b_val, c_val)

            # Curve evaluation
            d_points = np.logspace(-3, 2.3, num=200) # 0.001 to ~200 mm
            y_points = rosin_rammler_curve(d_points, b_val, c_val)

        # -----------------------------------------------------
        # Layout: Visual Inspection (Images)
        # -----------------------------------------------------
        st.markdown('<div class="section-title">📷 Image Inspection & Crop ROI</div>', unsafe_allow_html=True)
        col_img1, col_img2 = st.columns([1.2, 1])

        with col_img1:
            st.image(image_to_process, caption=f"Original Photograph ({orig_w} × {orig_h} px)", use_container_width=True)
            st.caption(f"**Source:** {image_source_label} | **Profile:** {device_desc}")

        with col_img2:
            st.image(cropped_pil, caption="Rescaled & Center-Cropped Input to CNN (400 × 400 px)", use_container_width=True)
            st.info(f"**Active Scale Factor:** `{active_scale:.4f}` | Resized to `{resized_dims[0]}×{resized_dims[1]}` px before center crop.")

        st.markdown("---")

        # -----------------------------------------------------
        # Layout: Key Predicted Parameters & Coefficients
        # -----------------------------------------------------
        st.markdown('<div class="section-title">📊 Predicted Rosin-Rammler & Geotechnical Parameters</div>', unsafe_allow_html=True)
        
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
        with m_col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Scale (b)</div>
                <div class="metric-value">{b_val:.3f} <span style="font-size: 0.9rem; font-weight: normal;">mm</span></div>
                <div class="metric-sub">Rosin-Rammler b</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Gradation (c)</div>
                <div class="metric-value">{c_val:.3f}</div>
                <div class="metric-sub">Uniformity exponent</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">D₁₀ (Effective)</div>
                <div class="metric-value">{d10:.3f} <span style="font-size: 0.9rem; font-weight: normal;">mm</span></div>
                <div class="metric-sub">10% Passing</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">D₅₀ (Median)</div>
                <div class="metric-value">{d50:.3f} <span style="font-size: 0.9rem; font-weight: normal;">mm</span></div>
                <div class="metric-sub">50% Passing</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col5:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Cu (Uniformity)</div>
                <div class="metric-value">{cu:.2f}</div>
                <div class="metric-sub">D₆₀ / D₁₀</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col6:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Cc (Curvature)</div>
                <div class="metric-value">{cc:.2f}</div>
                <div class="metric-sub">(D₃₀)² / (D₁₀·D₆₀)</div>
            </div>
            """, unsafe_allow_html=True)

        # -----------------------------------------------------
        # Layout: Particle Size Distribution Curves
        # -----------------------------------------------------
        st.markdown('<div class="section-title">📈 Particle Size Distribution Curves</div>', unsafe_allow_html=True)

        tab_interactive, tab_publication, tab_fractions = st.tabs([
            "✨ Interactive Plotly Curve",
            "📄 Scientific Matplotlib Plot",
            "🧩 Soil Fractions Breakdown"
        ])

        with tab_interactive:
            plotly_fig = create_plotly_psd(b_val, c_val, d_points, y_points, d10, d30, d50, d60)
            st.plotly_chart(plotly_fig, use_container_width=True)

        with tab_publication:
            mpl_fig = create_matplotlib_psd(b_val, c_val, d_points, y_points)
            st.pyplot(mpl_fig)
            
            # Matplotlib image download
            buf = io.BytesIO()
            mpl_fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            st.download_button(
                label="📥 Download Publication Figure (PNG, 300 DPI)",
                data=buf.getvalue(),
                file_name="soil_psd_curve.png",
                mime="image/png"
            )

        with tab_fractions:
            st.write("#### Geotechnical Grain-Size Classification Breakdown")
            f_col1, f_col2 = st.columns([1, 1])
            with f_col1:
                fraction_df = pd.DataFrame([
                    {"Fraction": k, "Percentage (%)": round(v, 2)}
                    for k, v in fractions.items()
                ])
                st.dataframe(fraction_df, hide_index=True, use_container_width=True)

            with f_col2:
                # Progress bars for each fraction
                for frac_name, frac_val in fractions.items():
                    st.write(f"**{frac_name}**: `{frac_val:.1f}%`")
                    st.progress(min(max(float(frac_val) / 100.0, 0.0), 1.0))

        st.markdown("---")

        # -----------------------------------------------------
        # Data Export Options
        # -----------------------------------------------------
        st.markdown('<div class="section-title">💾 Export Data & Report</div>', unsafe_allow_html=True)
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            # CSV download of sampled PSD points
            csv_df = pd.DataFrame({
                "particle_size_d_mm": np.round(d_points, 5),
                "fraction_finer_y": np.round(y_points, 5),
                "percent_finer": np.round(y_points * 100, 2)
            })
            csv_data = csv_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Curve Data Points (CSV)",
                data=csv_data,
                file_name="predicted_psd_curve.csv",
                mime="text/csv",
                use_container_width=True
            )

        with exp_col2:
            # JSON report
            report = {
                "source": image_source_label,
                "profile": device_desc,
                "scale_factor": active_scale,
                "parameters": {
                    "rosin_rammler_b_mm": round(b_val, 4),
                    "rosin_rammler_c": round(c_val, 4)
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
                "soil_fractions_percent": {k: round(v, 2) for k, v in fractions.items()}
            }
            json_str = json.dumps(report, indent=2)
            st.download_button(
                label="📥 Download Full Analysis Report (JSON)",
                data=json_str,
                file_name="soil_psd_analysis_report.json",
                mime="application/json",
                use_container_width=True
            )

else:
    # Empty State Guidance
    st.info("👆 Please upload a soil image from your computer or click **'Load Sample Soil Image'** in the sidebar to test immediately.")

