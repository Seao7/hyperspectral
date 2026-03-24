import os
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from i18n import t

ANALYSIS_SPACE = "."
REGISTRY_FILE  = "name_registry.json"
WAVELENGTHS    = list(range(350, 1051, 5))

# Replace this in both pages:
if "lang" not in st.session_state:
    st.session_state.lang = "en"
lang = st.session_state.lang

# With this:
if "lang" not in st.session_state:
    st.session_state.lang = "en"

with st.sidebar:
    st.title("⚙️ Settings")
    if st.button("🌐  EN / 日本語"):
        st.session_state.lang = "ja" if st.session_state.lang == "en" else "en"
    st.caption(f"{'English' if st.session_state.lang == 'en' else '日本語'}")
    st.markdown("---")
    st.markdown("**Pages**")
    st.markdown("🏠 Home — Browse & inspect files")
    st.markdown("📈 Spectra Comparison — Compare scans")
    st.markdown("🖼️ RGB Gallery — View all composites")

lang = st.session_state.lang

@st.cache_data
def load_registry():
    if not os.path.exists(REGISTRY_FILE):
        return {}
    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def get_display_name(stem, registry, lang):
    entry = registry.get(stem, {})
    en    = entry.get("display_en", stem)
    ja    = entry.get("display_ja", stem)
    return f"{ja}  /  {en}" if lang == "ja" else f"{en}  /  {ja}"

registry = load_registry()

st.title("📈  " + t("spectra_comparison", lang))

with st.expander("ℹ️  How to use this page" if lang == "en" else "ℹ️  使い方", expanded=False):
    if lang == "en":
        st.markdown("""
        1. Use the **multiselect box** below to choose any combination of scans across all folders.
        2. Each selected scan appears as a **separate line** on the chart.
        3. Hover over any line to see exact reflectance values at a given wavelength.
        4. Click a legend entry to **show/hide** individual scans on the chart.
        5. Use the Plotly toolbar (top right of chart) to zoom, pan, or download the chart.
        """)
    else:
        st.markdown("""
        1. 下の**マルチセレクトボックス**を使って、全フォルダからスキャンを選択してください。
        2. 選択した各スキャンはチャート上に**別々の線**として表示されます。
        3. 線の上にカーソルを置くと、指定した波長の正確な反射率が表示されます。
        4. 凡例のエントリをクリックすると、個別のスキャンを**表示/非表示**にできます。
        5. Plotlyツールバー（チャートの右上）でズーム、パン、ダウンロードができます。
        """)

st.markdown("---")

all_options = {}
for folder in sorted(os.listdir(ANALYSIS_SPACE)):
    folder_path = os.path.join(ANALYSIS_SPACE, folder)
    mean_dir    = os.path.join(folder_path, "mean_spectrums")
    if not os.path.isdir(mean_dir):
        continue
    for f in sorted(os.listdir(mean_dir)):
        if f.endswith("_spectra.npy"):
            stem  = f.replace("_spectra.npy", "")
            label = f"{folder}  ›  {get_display_name(stem, registry, lang)}"
            all_options[label] = (folder_path, stem)

selected_labels = st.multiselect(t("select_files", lang), list(all_options.keys()))

if not selected_labels:
    st.info(t("no_selection", lang))
    st.stop()

fig = go.Figure()
for label in selected_labels:
    folder_path, stem = all_options[label]
    spectra_path = os.path.join(folder_path, "mean_spectrums", stem + "_spectra.npy")
    if os.path.exists(spectra_path):
        spectra = np.load(spectra_path)
        fig.add_trace(go.Scatter(
            x=WAVELENGTHS[:len(spectra)], y=spectra,
            mode="lines", name=label, line=dict(width=2)
        ))

fig.update_layout(
    xaxis_title=t("wavelength", lang),
    yaxis_title=t("reflectance", lang),
    height=500,
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(color="white"),
    legend=dict(bgcolor="#0e1117")
)
st.plotly_chart(fig, use_container_width=True)
