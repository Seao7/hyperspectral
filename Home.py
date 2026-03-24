import os
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from i18n import t

ANALYSIS_SPACE = "."
REGISTRY_FILE  = "name_registry.json"
WAVELENGTHS    = list(range(350, 1051, 5))

st.set_page_config(
    page_title="Hyperspectral Viewer",
    page_icon="🌿",
    layout="wide"
)

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

@st.cache_data
def find_data_folders(root):
    folders = []
    for folder in sorted(os.listdir(root)):
        path = os.path.join(root, folder)
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, "mean_spectrums")):
            folders.append((folder, path))
    return folders

def get_display_name(stem, registry, lang):
    entry = registry.get(stem, {})
    en    = entry.get("display_en", stem)
    ja    = entry.get("display_ja", stem)
    return f"{ja}  /  {en}" if lang == "ja" else f"{en}  /  {ja}"

def get_scan_stems(folder_path):
    mean_dir = os.path.join(folder_path, "mean_spectrums")
    return sorted([
        f.replace("_spectra.npy", "")
        for f in os.listdir(mean_dir)
        if f.endswith("_spectra.npy")
    ])

@st.cache_data
def load_spectra(spectra_path):
    return np.load(spectra_path)

# ── PAGE HEADER ───────────────────────────────────────────────────────────────
st.title("🏠  " + t("app_title", lang))

# Instructions
with st.expander("ℹ️  How to use this page" if lang == "en" else "ℹ️  使い方", expanded=False):
    if lang == "en":
        st.markdown("""
        1. **Select a folder** from the left panel — each folder corresponds to a data collection batch.
        2. **Select a file** from the list below the folder — file names are shown in both English and Japanese.
        3. The **RGB composite** and **mean spectral graph** for the selected file will appear on the right.
        4. The spectral graph shows average reflectance across all pixels, plotted against wavelength (350–1050 nm).
        5. Use the sidebar to switch language or navigate to other pages.
        """)
    else:
        st.markdown("""
        1. 左パネルから**フォルダを選択**してください。各フォルダはデータ収集バッチに対応しています。
        2. フォルダ下のリストから**ファイルを選択**してください。ファイル名は英語と日本語で表示されます。
        3. 選択したファイルの**RGBコンポジット**と**平均スペクトルグラフ**が右側に表示されます。
        4. スペクトルグラフは、全ピクセルの平均反射率を波長（350〜1050 nm）に対してプロットしています。
        5. サイドバーで言語を切り替えたり、他のページに移動できます。
        """)

st.markdown("---")

registry     = load_registry()
data_folders = find_data_folders(ANALYSIS_SPACE)

if not data_folders:
    st.warning(t("no_folders", lang))
    st.stop()

left, right = st.columns([1, 2.5])

with left:
    st.subheader(t("file_browser", lang))

    folder_labels        = [name for name, _ in data_folders]
    selected_folder_name = st.selectbox(t("select_folder", lang), folder_labels)
    selected_folder_path = dict(data_folders)[selected_folder_name]

    stems = get_scan_stems(selected_folder_path)

    if not stems:
        st.info(t("no_files", lang))
        st.stop()

    stem_label_map = {get_display_name(s, registry, lang): s for s in stems}
    selected_label = st.radio(t("select_file", lang), list(stem_label_map.keys()))
    selected_stem  = stem_label_map[selected_label]

with right:
    rgb_path     = os.path.join(selected_folder_path, "rgb_composites", selected_stem + ".jpg")
    spectra_path = os.path.join(selected_folder_path, "mean_spectrums", selected_stem + "_spectra.npy")

    st.subheader(get_display_name(selected_stem, registry, lang))

    img_col, spec_col = st.columns(2)

    with img_col:
        st.caption(t("rgb_image", lang))
        if os.path.exists(rgb_path):
            st.image(Image.open(rgb_path), use_container_width=True)
        else:
            st.warning("RGB image not found.")

    with spec_col:
        st.caption(t("mean_spectra", lang))
        if os.path.exists(spectra_path):
            spectra = load_spectra(spectra_path)
            x_axis  = WAVELENGTHS[:len(spectra)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_axis, y=spectra,
                mode="lines",
                line=dict(color="#2ecc71", width=2),
                name=selected_label
            ))
            fig.update_layout(
                xaxis_title=t("wavelength", lang),
                yaxis_title=t("reflectance", lang),
                margin=dict(l=20, r=20, t=20, b=40),
                height=350,
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font=dict(color="white")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Spectra file not found.")
