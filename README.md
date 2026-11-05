---
title: Soil Particle Size Distribution (PSD) Analyzer
emoji: 🌱
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.56.0
app_file: app.py
pinned: false
license: mit
short_description: Soil particle size distribution curve predictor from smartphone photographs
---

# 🌱 Soil Particle Size Distribution (PSD) Analyzer

A deep learning-powered web application that estimates the **Particle Size Distribution (PSD) curve** and geotechnical gradation parameters of soil directly from close-range smartphone photographs.

---

## 🌟 Key Features

- **🧠 Deep Learning Inference**: Uses a pre-trained MobileNet-based Convolutional Neural Network (`CNN.h5`) to predict the Rosin-Rammler parameters $b$ (scale) and $c$ (shape).
- **📈 Interactive & Publication Plots**:
  - **Interactive Plotly Graph**: Dynamic hover tooltips, semilogarithmic axes, shaded soil classification zones (Clay, Silt, Sand, Gravel, Cobbles), and marked characteristic diameters ($D_{10}, D_{30}, D_{50}, D_{60}$).
  - **Scientific Matplotlib Figure**: High-resolution, publication-ready figure styled according to standard geotechnical laboratory sieve analysis standards.
- **📊 Comprehensive Geotechnical Metrics**:
  - **Rosin-Rammler Parameters**: Scale parameter $b$ (mm) and uniformity shape exponent $c$.
  - **Characteristic Diameters**: $D_{10}$ (effective diameter), $D_{30}$, $D_{50}$ (median diameter), $D_{60}$.
  - **Gradation Coefficients**: Coefficient of Uniformity ($C_u = D_{60}/D_{10}$) and Coefficient of Curvature ($C_c = D_{30}^2 / (D_{10} \cdot D_{60})$).
  - **Soil Classification Breakdown**: Quantified percentages for Clay, Silt, Sand, Gravel, and Cobbles/Boulders.
- **📱 Smartphone Calibration Profiles**:
  - Automatic resolution detection and calibrated focal scaling for popular devices (Motorola Edge, Samsung Galaxy A52, Apple iPhone 14).
  - Custom scaling factor input for arbitrary camera resolutions.
- **💾 One-Click Data Export**:
  - Export the sampled curve points to CSV.
  - Download high-resolution publication figures (PNG, 300 DPI).
  - Download structured analysis reports in JSON format.
- **⚡ Instant Testing**: Built-in realistic soil sample photograph to test the full pipeline with a single click.

---

## 🔬 Mathematical Methodology

### 1. Rosin-Rammler Distribution Model
The cumulative passing fraction $y(d)$ (fraction of soil mass finer than particle diameter $d$ in mm) is modeled as:

$$y(d) = 1 - \exp\left(-\left(\frac{d}{b}\right)^c\right)$$

Where:
- $d$: Particle diameter (mm)
- $b$: Characteristic scale parameter (mm)
- $c$: Gradation shape parameter (slope of the curve on a Weibull plot)

### 2. Characteristic Particle Diameters
For any cumulative passing fraction $p \in (0, 1)$:

$$D_p = b \cdot \left(-\ln(1 - p)\right)^{1/c}$$

- **Effective size ($D_{10}$)**: $p = 0.10 \implies D_{10} = b \cdot (-\ln 0.90)^{1/c}$
- **Median size ($D_{50}$)**: $p = 0.50 \implies D_{50} = b \cdot (\ln 2)^{1/c}$

### 3. Geotechnical Gradation Indices
- **Uniformity Coefficient**: $C_u = \frac{D_{60}}{D_{10}}$
- **Curvature Coefficient**: $C_c = \frac{(D_{30})^2}{D_{10} \cdot D_{60}}$

### 4. Soil Particle Fractions
Standard geotechnical sieve boundaries (mm):
| Fraction | Size Range ($d$) |
| :--- | :--- |
| **Clay (Cl)** | $d < 0.002\text{ mm}$ |
| **Silt (Si)** | $0.002\text{ mm} \le d < 0.063\text{ mm}$ |
| **Sand (Sa)** | $0.063\text{ mm} \le d < 2.0\text{ mm}$ |
| **Gravel (Gr)** | $2.0\text{ mm} \le d < 63.0\text{ mm}$ |
| **Cobbles / Boulders (Bo)** | $d \ge 63.0\text{ mm}$ |

---

## 🚀 Getting Started

### Prerequisites & Conda Environment

The application runs seamlessly with your existing Conda environment (`dl_clean`):

```bash
# Activate your conda environment
conda activate dl_clean
```

### Installation of Dependencies

If any package needs to be updated:

```bash
pip install -r requirements.txt
```

### Running the Application

You can launch the application in any of the following ways:

#### Option 1: Double-Click Launcher (Windows)
Double-click `run_app.bat` or run `.\run_app.ps1` in PowerShell.

#### Option 2: Command Line
```bash
conda activate dl_clean
streamlit run app.py
```

The web interface will automatically open in your default browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```text
├── CNN.h5              # Pre-trained deep learning regression model
├── app.py              # Main Streamlit web application frontend & inference engine
├── requirements.txt    # Python runtime dependencies
├── run_app.bat         # Windows batch one-click launcher
├── run_app.ps1         # PowerShell one-click launcher
├── sample_images/      # Sample soil photographs for instant demonstration
│   └── soil_sample.jpg # Built-in high-resolution soil sample photograph
└── README.md           # Documentation & project manual
```

---

## 💡 Usage Tips

1. **Photo Capture**: Capture a close-range, top-down photograph of your soil sample under diffuse, shadow-free natural lighting.
2. **Device Profile**: Choose your phone model from the sidebar profile dropdown or allow auto-detection based on image resolution.
3. **Interactive Analysis**: Hover over points on the Plotly graph to examine particle sizes and percentages passing at any diameter.
4. **Export**: Use the export buttons at the bottom to save your curve data to CSV or download high-resolution figures for reports.