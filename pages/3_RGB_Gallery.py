import os
import json
import streamlit as st
from PIL import Image
from i18n import t

ANALYSIS_SPACE = "."
REGISTRY_FILE  = "name_registry.json"
COLS           = 3

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

st.title("🖼️  " + t("all_rgb", lang))

with st.expander("ℹ️  How to use this page" if lang == "en" else "ℹ️  使い方", expanded=False):
    if lang == "en":
        st.markdown("""
        1. All RGB composite images are displayed here, grouped by data folder.
        2. Each image caption shows the scan name in both English and Japanese.
        3. Click any image to expand it for a closer look.
        4. To inspect the spectra of a specific scan, go back to the **Home** page and select it there.
        """)
    else:
        st.markdown("""
        1. 全てのRGBコンポジット画像がデータフォルダごとにグループ化されて表示されます。
        2. 各画像のキャプションにはスキャン名が英語と日本語で表示されます。
        3. 画像をクリックすると拡大表示されます。
        4. 特定のスキャンのスペクトルを確認するには、**ホーム**ページに戻って選択してください。
        """)

st.markdown("---")

for folder in sorted(os.listdir(ANALYSIS_SPACE)):
    folder_path = os.path.join(ANALYSIS_SPACE, folder)
    rgb_dir     = os.path.join(folder_path, "rgb_composites")
    if not os.path.isdir(rgb_dir):
        continue

    images = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
    if not images:
        continue

    st.subheader(f"{t('folder_label', lang)}: {folder}")
    cols = st.columns(COLS)

    for i, img_file in enumerate(images):
        stem    = img_file.replace(".jpg", "")
        caption = get_display_name(stem, registry, lang)
        with cols[i % COLS]:
            st.image(
                Image.open(os.path.join(rgb_dir, img_file)),
                caption=caption,
                use_container_width=True
            )

    st.markdown("---")
