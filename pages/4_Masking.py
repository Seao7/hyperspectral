import os
import io
import json
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from skimage.draw import polygon2mask
from i18n import t

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Masking", layout="wide")

ANALYSIS_SPACE = "."
REGISTRY_FILE  = "name_registry.json"
MAX_CANVAS_W   = 900

# ── SESSION STATE ─────────────────────────────────────────────────────────────
defaults = {
    "lang":          "en",
    "shapes":        [],
    "pending_json":  None,
    "auto_mask":     None,
    "stroke_width":  2,
    "stroke_color":  "#ff0000",
    "fill_color":    "#ff0000",
    "fill_alpha":    0.2,
    "draw_mode":     "Polygon",
    "last_stem":     None,
    "img_cache_key": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    if st.button("🌐  EN / 日本語"):
        st.session_state.lang = "ja" if st.session_state.lang == "en" else "en"
    st.caption("English" if st.session_state.lang == "en" else "日本語")
    st.markdown("---")
    st.markdown("**Pages**")
    st.markdown("🏠 Home — Browse & inspect files")
    st.markdown("📈 Spectra Comparison — Compare scans")
    st.markdown("🖼️ RGB Gallery — View all composites")
    st.markdown("📐 Masking — Create & edit masks")

lang = st.session_state.lang

# ── HELPERS ───────────────────────────────────────────────────────────────────
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

def find_data_folders(root):
    folders = []
    for folder in sorted(os.listdir(root)):
        path = os.path.join(root, folder)
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, "rgb_composites")):
            folders.append((folder, path))
    return folders

def get_stems(folder_path):
    rgb_dir = os.path.join(folder_path, "rgb_composites")
    return sorted([f.replace(".jpg", "") for f in os.listdir(rgb_dir) if f.endswith(".jpg")])

def get_mask_path(folder_path, stem):
    return os.path.join(folder_path, "masks", stem + "_mask.npy")

def load_rgb(folder_path, stem):
    path = os.path.join(folder_path, "rgb_composites", stem + ".jpg")
    return np.array(Image.open(path).convert("RGB"))

def scale_image(img_np, max_w):
    H, W   = img_np.shape[:2]
    scale  = min(1.0, max_w / W)
    new_w  = int(W * scale)
    new_h  = int(H * scale)
    img_rs = Image.fromarray(img_np).resize((new_w, new_h), resample=Image.BILINEAR)
    return np.array(img_rs), new_w, new_h, W / new_w, H / new_h

def save_mask(folder_path, stem, mask_bool):
    masks_dir = os.path.join(folder_path, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    np.save(get_mask_path(folder_path, stem), mask_bool)

# ── GRABCUT ───────────────────────────────────────────────────────────────────
def run_grabcut(img_np, rect, iterations=5):
    H, W      = img_np.shape[:2]
    mask      = np.zeros((H, W), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(img_np, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), True, False)

# ── MANUAL MASK HELPERS ───────────────────────────────────────────────────────
def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b   = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (r, g, b, int(alpha * 255))

def parse_polygon_points(obj):
    if isinstance(obj, dict) and isinstance(obj.get("path"), list):
        pts = []
        for seg in obj["path"]:
            if not isinstance(seg, list) or len(seg) < 1:
                continue
            if seg[0] in ("M", "L") and len(seg) >= 3:
                try:
                    pts.append([float(seg[1]), float(seg[2])])
                except Exception:
                    continue
        if len(pts) >= 3:
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            return pts
    pts_local = obj.get("points", [])
    if isinstance(pts_local, list) and len(pts_local) >= 3:
        left_o  = float(obj.get("left",   0))
        top_o   = float(obj.get("top",    0))
        scale_x = float(obj.get("scaleX", 1.0))
        scale_y = float(obj.get("scaleY", 1.0))
        pts     = []
        for p in pts_local:
            if isinstance(p, dict):
                pts.append([float(p.get("x", 0)) * scale_x + left_o,
                             float(p.get("y", 0)) * scale_y + top_o])
        if len(pts) >= 3:
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            return pts
    return None

def rasterize_shapes(H, W, shapes):
    mask = np.zeros((H, W), dtype=np.uint8)
    for s in shapes:
        if s["type"] == "polygon":
            pts = np.array(s["points"], dtype=np.float32)
            if len(pts) > 2 and np.allclose(pts[0], pts[-1]):
                pts = pts[:-1]
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            rr = polygon2mask((H, W), np.stack([pts[:, 1], pts[:, 0]], axis=1))
            mask[rr] = 255
        elif s["type"] == "circle":
            cx, cy = s["center"]
            r      = s["radius"]
            cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
    return mask

# ── PAGE ──────────────────────────────────────────────────────────────────────
st.title("📐  Masking" if lang == "en" else "📐  マスク生成")

registry     = load_registry()
data_folders = find_data_folders(ANALYSIS_SPACE)

if not data_folders:
    st.warning("No processed folders found. Run process.py first." if lang == "en"
               else "処理済みフォルダが見つかりません。process.pyを実行してください。")
    st.stop()

# ── FILE SELECTOR ─────────────────────────────────────────────────────────────
sel_col1, sel_col2 = st.columns(2)
with sel_col1:
    folder_labels        = [name for name, _ in data_folders]
    selected_folder_name = st.selectbox(
        "📁 Select Folder" if lang == "en" else "📁 フォルダを選択",
        folder_labels
    )
    selected_folder_path = dict(data_folders)[selected_folder_name]

with sel_col2:
    stems          = get_stems(selected_folder_path)
    stem_label_map = {get_display_name(s, registry, lang): s for s in stems}
    selected_label = st.selectbox(
        "🗂️ Select File" if lang == "en" else "🗂️ ファイルを選択",
        list(stem_label_map.keys())
    )
    selected_stem = stem_label_map[selected_label]

# ── Clear state when file changes ────────────────────────────────────────────
if st.session_state.last_stem != selected_stem:
    st.session_state.shapes       = []
    st.session_state.auto_mask    = None
    st.session_state.pending_json = None
    st.session_state.last_stem    = selected_stem

# ── Load and cache display image ──────────────────────────────────────────────
img_np    = load_rgb(selected_folder_path, selected_stem)
H, W      = img_np.shape[:2]
cache_key = f"{selected_folder_name}_{selected_stem}"

if st.session_state.img_cache_key != cache_key or "img_disp_pil" not in st.session_state:
    arr, cw_v, ch_v, sx_v, sy_v   = scale_image(img_np, MAX_CANVAS_W)
    pil_img                        = Image.fromarray(arr)
    st.session_state.img_cache_key = cache_key
    st.session_state.img_disp_arr  = arr
    st.session_state.img_disp_pil  = pil_img
    st.session_state.img_disp_cw   = cw_v
    st.session_state.img_disp_ch   = ch_v
    st.session_state.img_disp_sx   = sx_v
    st.session_state.img_disp_sy   = sy_v

img_disp_pil = Image.fromarray(st.session_state.img_disp_arr)
cw   = st.session_state.img_disp_cw
ch   = st.session_state.img_disp_ch
sx   = st.session_state.img_disp_sx
sy   = st.session_state.img_disp_sy

# ── Existing mask indicator ───────────────────────────────────────────────────
mask_path     = get_mask_path(selected_folder_path, selected_stem)
existing_mask = np.load(mask_path) if os.path.exists(mask_path) else None

if existing_mask is not None:
    st.success("✅ A saved mask exists for this file." if lang == "en"
               else "✅ このファイルには保存済みマスクがあります。")
    with st.expander("👁️ View saved mask" if lang == "en" else "👁️ 保存済みマスクを表示",
                     expanded=False):
        overlay = img_np.copy()
        overlay[existing_mask] = (
            overlay[existing_mask] * 0.5 + np.array([0, 200, 0]) * 0.5
        ).astype(np.uint8)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Original RGB")
            st.image(img_np, use_container_width=True)
        with c2:
            st.caption("Saved mask (green = foreground)" if lang == "en"
                       else "保存済みマスク（緑 = 前景）")
            st.image(overlay, use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
# tab_auto, tab_manual = st.tabs([
#     "🤖 Auto Mask (GrabCut)" if lang == "en" else "🤖 自動マスク (GrabCut)",
#     "✏️ Manual Mask"         if lang == "en" else "✏️ 手動マスク"
# ])

# # ════════════════════════════════════════════════════════════════════════════
# # TAB 1 — AUTO MASK
# # ════════════════════════════════════════════════════════════════════════════
# with tab_auto:
#     with st.expander("ℹ️ How to use" if lang == "en" else "ℹ️ 使い方", expanded=False):
#         if lang == "en":
#             st.markdown("""
#             1. Draw a rectangle around the foreground sample on the canvas.
#             2. Click **Run GrabCut** — the algorithm separates foreground from background.
#             3. Review the mask preview. Green overlay = foreground kept.
#             4. Click **Save Mask** to write it to `masks/`, or **Discard** to ignore.
#             """)
#         else:
#             st.markdown("""
#             1. キャンバス上で前景サンプルの周囲に矩形を描いてください。
#             2. **GrabCutを実行**をクリックすると、前景と背景が分離されます。
#             3. マスクのプレビューを確認してください。緑のオーバーレイ＝前景として保持。
#             4. **マスクを保存**で`masks/`に書き込むか、**破棄**で無視します。
#             """)

#     st.caption("🖱️ Draw a rectangle around the foreground sample." if lang == "en"
#                else "🖱️ 前景サンプルの周囲に矩形を描いてください。")

#     canvas_auto = st_canvas(
#         fill_color       = "rgba(255, 0, 0, 0.15)",
#         stroke_width     = 2,
#         stroke_color     = "#ff0000",
#         background_image = img_disp_pil,
#         update_streamlit = True,
#         height           = ch,
#         width            = cw,
#         drawing_mode     = "rect",
#         key              = f"canvas_auto_{selected_stem}",
#         display_toolbar  = True,
#     )

#     if st.button("▶️ Run GrabCut" if lang == "en" else "▶️ GrabCutを実行"):
#         rect_to_use = None
#         data        = canvas_auto.json_data

#         if data and "objects" in data and len(data["objects"]) > 0:
#             obj = data["objects"][-1]
#             if obj.get("type") == "rect":
#                 left   = int(float(obj.get("left",   0)) * sx)
#                 top    = int(float(obj.get("top",    0)) * sy)
#                 width  = int(float(obj.get("width",  0)) * sx * float(obj.get("scaleX", 1)))
#                 height = int(float(obj.get("height", 0)) * sy * float(obj.get("scaleY", 1)))
#                 if width > 10 and height > 10:
#                     rect_to_use = (
#                         max(0, left),
#                         max(0, top),
#                         min(width,  W - left),
#                         min(height, H - top)
#                     )

#         if rect_to_use is None:
#             st.warning("Please draw a rectangle on the canvas first." if lang == "en"
#                        else "まずキャンバスに矩形を描いてください。")
#         else:
#             with st.spinner("Running GrabCut..." if lang == "en" else "GrabCut実行中..."):
#                 st.session_state.auto_mask = run_grabcut(img_np, rect_to_use)
#             st.success("Done! Review the mask below." if lang == "en"
#                        else "完了！以下のマスクを確認してください。")

#     if st.session_state.auto_mask is not None:
#         mask = st.session_state.auto_mask
#         c1, c2 = st.columns(2)
#         with c1:
#             st.caption("Original RGB")
#             st.image(img_np, use_container_width=True)
#         with c2:
#             st.caption("Foreground mask (green overlay)" if lang == "en"
#                        else "前景マスク（緑のオーバーレイ）")
#             overlay       = img_np.copy()
#             overlay[mask] = (overlay[mask] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
#             st.image(overlay, use_container_width=True)

#         save_col, discard_col = st.columns(2)
#         with save_col:
#             if st.button("💾 Save Mask" if lang == "en" else "💾 マスクを保存",
#                          type="primary"):
#                 save_mask(selected_folder_path, selected_stem, mask)
#                 st.success(f"Saved to masks/{selected_stem}_mask.npy ✅")
#         with discard_col:
#             if st.button("🗑️ Discard" if lang == "en" else "🗑️ 破棄"):
#                 st.session_state.auto_mask = None
#                 st.rerun()

# # ════════════════════════════════════════════════════════════════════════════
# # TAB 2 — MANUAL MASK
# # ════════════════════════════════════════════════════════════════════════════
# with tab_manual:
#     with st.expander("ℹ️ How to use" if lang == "en" else "ℹ️ 使い方", expanded=False):
#         if lang == "en":
#             st.markdown("""
#             1. Choose **Polygon** or **Circle** tool from Drawing Settings below.
#             2. Draw shapes over the foreground area. Right-click to close a polygon.
#             3. Click **Commit Shape** to register each shape.
#             4. Add as many shapes as needed — they are combined automatically.
#             5. Click **Generate & Save Mask** — this becomes the final mask for this file.
#             6. If an auto mask already existed, this will replace it.
#             """)
#         else:
#             st.markdown("""
#             1. 下の描画設定から**ポリゴン**または**円**ツールを選択してください。
#             2. 前景領域の上に図形を描きます。ポリゴンを閉じるには右クリックしてください。
#             3. **図形をコミット**をクリックして各図形を登録してください。
#             4. 必要な数だけ図形を追加できます — 自動的に結合されます。
#             5. **マスクを生成して保存**をクリックすると、このファイルの最終マスクになります。
#             6. 自動マスクが既に存在する場合は上書きされます。
#             """)

#     fill_rgba = hex_to_rgba(st.session_state.fill_color, st.session_state.fill_alpha)
#     tool_mode = "polygon" if st.session_state.draw_mode == "Polygon" else "circle"

#     st.caption("Draw shapes over the foreground. Commit each shape before drawing the next." if lang == "en"
#                else "前景の上に図形を描いてください。次の図形を描く前に各図形をコミットしてください。")

#     canvas_manual = st_canvas(
#         fill_color       = fill_rgba,
#         stroke_width     = st.session_state.stroke_width,
#         stroke_color     = st.session_state.stroke_color,
#         background_image = img_disp_pil,
#         update_streamlit = True,
#         height           = ch,
#         width            = cw,
#         drawing_mode     = tool_mode,
#         key              = f"canvas_manual_{selected_stem}",
#         display_toolbar  = True,
#     )
#     st.session_state.pending_json = canvas_manual.json_data

#     # ── Shapes list ───────────────────────────────────────────────────────────
#     if st.session_state.shapes:
#         st.caption(f"{'Committed shapes' if lang == 'en' else 'コミット済みの図形'}: "
#                    f"{len(st.session_state.shapes)}")
#         for i, s in enumerate(st.session_state.shapes):
#             if s["type"] == "polygon":
#                 n      = len(s["points"])
#                 n_show = n - 1 if (n > 2 and s["points"][0] == s["points"][-1]) else n
#                 st.write(f"  {i+1}. Polygon — {n_show} vertices")
#             else:
#                 st.write(f"  {i+1}. Circle — center {tuple(map(int, s['center']))}, "
#                          f"r={int(s['radius'])}")
#         if st.button("🗑️ Clear all shapes" if lang == "en" else "🗑️ 全図形をクリア"):
#             st.session_state.shapes = []
#             st.rerun()
#     else:
#         st.caption("No shapes committed yet." if lang == "en"
#                    else "まだ図形がコミットされていません。")

#     # ── Commit ────────────────────────────────────────────────────────────────
#     if st.button("📌 Commit Shape" if lang == "en" else "📌 図形をコミット"):
#         data = st.session_state.pending_json
#         if data and "objects" in data and len(data["objects"]) > 0:
#             obj      = data["objects"][-1]
#             obj_type = obj.get("type", "")

#             if obj_type in ("polygon", "path"):
#                 abs_pts = parse_polygon_points(obj)
#                 if abs_pts and len(abs_pts) >= 3:
#                     mapped = [[p[0] * sx, p[1] * sy] for p in abs_pts]
#                     st.session_state.shapes.append({"type": "polygon", "points": mapped})
#                     st.rerun()
#                 else:
#                     st.warning("Polygon not closed. Right-click to close, then Commit." if lang == "en"
#                                else "ポリゴンが閉じていません。右クリックで閉じてからコミットしてください。")
#             elif obj_type == "circle":
#                 left_o  = float(obj.get("left",   0))
#                 top_o   = float(obj.get("top",    0))
#                 scale_x = float(obj.get("scaleX", 1.0))
#                 radius  = float(obj.get("radius", obj.get("width", 0) / 2))
#                 cx      = (left_o + radius) * sx
#                 cy      = (top_o  + radius) * sy
#                 r       = radius * scale_x * (sx + sy) / 2.0
#                 st.session_state.shapes.append({"type": "circle", "center": [cx, cy], "radius": r})
#                 st.rerun()
#             else:
#                 st.warning(f"Unsupported shape type: {obj_type}" if lang == "en"
#                            else f"サポートされていない図形タイプ: {obj_type}")
#         else:
#             st.warning("No shape found to commit. Draw something first." if lang == "en"
#                        else "コミットする図形が見つかりません。まず図形を描いてください。")

#     # ── Generate & Save ───────────────────────────────────────────────────────
#     st.markdown("---")
#     if st.button("💾 Generate & Save Mask" if lang == "en" else "💾 マスクを生成して保存",
#                  type="primary",
#                  disabled=len(st.session_state.shapes) == 0):
#         mask_uint8 = rasterize_shapes(H, W, st.session_state.shapes)
#         mask_bool  = mask_uint8.astype(bool)
#         save_mask(selected_folder_path, selected_stem, mask_bool)
#         overlay              = img_np.copy()
#         overlay[mask_bool]   = (
#             overlay[mask_bool] * 0.5 + np.array([0, 200, 0]) * 0.5
#         ).astype(np.uint8)
#         st.image(overlay,
#                  caption="Saved mask — green = foreground" if lang == "en"
#                  else "保存済みマスク — 緑 = 前景",
#                  use_container_width=True)
#         st.success(f"Saved to masks/{selected_stem}_mask.npy ✅")
#         st.session_state.shapes = []

#     # ── Drawing settings ──────────────────────────────────────────────────────
#     with st.expander("🎨 Drawing Settings" if lang == "en" else "🎨 描画設定",
#                      expanded=False):
#         c1, c2, c3 = st.columns(3)
#         with c1:
#             st.session_state.draw_mode = st.radio(
#                 "Tool" if lang == "en" else "ツール",
#                 ["Polygon", "Circle"],
#                 index=0 if st.session_state.draw_mode == "Polygon" else 1
#             )
#             st.session_state.stroke_width = st.slider(
#                 "Outline width" if lang == "en" else "輪郭の幅",
#                 1, 10, st.session_state.stroke_width
#             )
#         with c2:
#             st.session_state.stroke_color = st.color_picker(
#                 "Outline color" if lang == "en" else "輪郭の色",
#                 st.session_state.stroke_color
#             )
#             st.session_state.fill_color = st.color_picker(
#                 "Fill color" if lang == "en" else "塗りつぶし色",
#                 st.session_state.fill_color
#             )
#         with c3:
#             st.session_state.fill_alpha = st.slider(
#                 "Fill opacity" if lang == "en" else "塗りつぶし透明度",
#                 0.0, 1.0, st.session_state.fill_alpha, 0.05
#             )

# ── Mode selector — replaces tabs ─────────────────────────────────────────
mode = st.radio(
    "Mode" if lang == "en" else "モード",
    ["🤖 Auto Mask (GrabCut)", "✏️ Manual Mask"],
    horizontal=True
)
st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# AUTO MASK
# ════════════════════════════════════════════════════════════════════════════
if mode == "🤖 Auto Mask (GrabCut)":
    with st.expander("ℹ️ How to use" if lang == "en" else "ℹ️ 使い方", expanded=False):
        if lang == "en":
            st.markdown("""
            1. Draw a rectangle around the foreground sample on the canvas.
            2. Click **Run GrabCut** — the algorithm separates foreground from background.
            3. Review the mask preview. Green overlay = foreground kept.
            4. Click **Save Mask** to write it to `masks/`, or **Discard** to ignore.
            """)
        else:
            st.markdown("""
            1. キャンバス上で前景サンプルの周囲に矩形を描いてください。
            2. **GrabCutを実行**をクリックすると、前景と背景が分離されます。
            3. マスクのプレビューを確認してください。緑のオーバーレイ＝前景として保持。
            4. **マスクを保存**で`masks/`に書き込むか、**破棄**で無視します。
            """)

    st.caption("🖱️ Draw a rectangle around the foreground sample." if lang == "en"
               else "🖱️ 前景サンプルの周囲に矩形を描いてください。")

    # ── Select entire image button ────────────────────────────────────────────
    if st.button("🖼️ Select Entire Image" if lang == "en" else "🖼️ 画像全体を選択"):
        st.session_state.auto_mask = run_grabcut(img_np, (1, 1, W - 2, H - 2))
        st.success("Done! Review the mask below." if lang == "en"
                else "完了！以下のマスクを確認してください。")

    st.caption("🖱️ Or draw one or more rectangles around the foreground sample." if lang == "en"
            else "🖱️ または、前景サンプルの周囲に矩形を描いてください。")

    canvas_auto = st_canvas(
        fill_color       = "rgba(255, 0, 0, 0.15)",
        stroke_width     = 2,
        stroke_color     = "#ff0000",
        background_image = img_disp_pil,
        update_streamlit = True,
        height           = ch,
        width            = cw,
        drawing_mode     = "rect",
        key              = f"canvas_auto_{selected_stem}",
        display_toolbar  = True,
    )

    if st.button("▶️ Run GrabCut" if lang == "en" else "▶️ GrabCutを実行"):
        data     = canvas_auto.json_data
        rects    = []

        if data and "objects" in data:
            for obj in data["objects"]:
                if obj.get("type") == "rect":
                    left   = float(obj.get("left",   0)) * sx
                    top    = float(obj.get("top",    0)) * sy
                    width  = float(obj.get("width",  0)) * sx * float(obj.get("scaleX", 1))
                    height = float(obj.get("height", 0)) * sy * float(obj.get("scaleY", 1))
                    if width > 10 and height > 10:
                        rects.append((left, top, left + width, top + height))  # x1,y1,x2,y2

        if not rects:
            st.warning("Please draw at least one rectangle, or use 'Select Entire Image'." if lang == "en"
                    else "矩形を少なくとも1つ描くか、「画像全体を選択」を使用してください。")
        else:
            # Merge all rectangles into one bounding box covering all of them
            x1 = max(0,     int(min(r[0] for r in rects)))
            y1 = max(0,     int(min(r[1] for r in rects)))
            x2 = min(W,     int(max(r[2] for r in rects)))
            y2 = min(H,     int(max(r[3] for r in rects)))
            merged_rect = (x1, y1, x2 - x1, y2 - y1)

            if len(rects) > 1:
                st.info(f"Merged {len(rects)} rectangles into one bounding box." if lang == "en"
                        else f"{len(rects)}個の矩形を1つのバウンディングボックスに統合しました。")

            with st.spinner("Running GrabCut..." if lang == "en" else "GrabCut実行中..."):
                st.session_state.auto_mask = run_grabcut(img_np, merged_rect)
            st.success("Done! Review the mask below." if lang == "en"
                    else "完了！以下のマスクを確認してください。")


    if st.session_state.auto_mask is not None:
        mask   = st.session_state.auto_mask
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Original RGB")
            st.image(img_np, use_container_width=True)
        with c2:
            st.caption("Foreground mask (green overlay)" if lang == "en"
                       else "前景マスク（緑のオーバーレイ）")
            overlay       = img_np.copy()
            overlay[mask] = (overlay[mask] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
            st.image(overlay, use_container_width=True)

        save_col, discard_col = st.columns(2)
        with save_col:
            if st.button("💾 Save Mask" if lang == "en" else "💾 マスクを保存", type="primary"):
                save_mask(selected_folder_path, selected_stem, mask)
                st.success(f"Saved to masks/{selected_stem}_mask.npy ✅")
        with discard_col:
            if st.button("🗑️ Discard" if lang == "en" else "🗑️ 破棄"):
                st.session_state.auto_mask = None
                st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# MANUAL MASK
# ════════════════════════════════════════════════════════════════════════════
elif mode == "✏️ Manual Mask":
    with st.expander("ℹ️ How to use" if lang == "en" else "ℹ️ 使い方", expanded=False):
        if lang == "en":
            st.markdown("""
            1. Choose **Polygon** or **Circle** tool from Drawing Settings below.
            2. Draw shapes over the foreground area. Right-click to close a polygon.
            3. Click **Commit Shape** to register each shape.
            4. Add as many shapes as needed — they are combined automatically.
            5. Click **Generate & Save Mask** — this becomes the final mask for this file.
            6. If an auto mask already existed, this will replace it.
            """)
        else:
            st.markdown("""
            1. 下の描画設定から**ポリゴン**または**円**ツールを選択してください。
            2. 前景領域の上に図形を描きます。ポリゴンを閉じるには右クリックしてください。
            3. **図形をコミット**をクリックして各図形を登録してください。
            4. 必要な数だけ図形を追加できます — 自動的に結合されます。
            5. **マスクを生成して保存**をクリックすると、このファイルの最終マスクになります。
            6. 自動マスクが既に存在する場合は上書きされます。
            """)

    fill_rgba = hex_to_rgba(st.session_state.fill_color, st.session_state.fill_alpha)
    tool_mode = "polygon" if st.session_state.draw_mode == "Polygon" else "circle"

    st.caption("Draw shapes over the foreground. Commit each shape before drawing the next." if lang == "en"
               else "前景の上に図形を描いてください。次の図形を描く前に各図形をコミットしてください。")

    canvas_manual = st_canvas(
        fill_color       = fill_rgba,
        stroke_width     = st.session_state.stroke_width,
        stroke_color     = st.session_state.stroke_color,
        background_image = img_disp_pil,
        update_streamlit = True,
        height           = ch,
        width            = cw,
        drawing_mode     = tool_mode,
        key              = f"canvas_manual_{selected_stem}",
        display_toolbar  = True,
    )
    st.session_state.pending_json = canvas_manual.json_data

    if st.session_state.shapes:
        st.caption(f"{'Committed shapes' if lang == 'en' else 'コミット済みの図形'}: "
                   f"{len(st.session_state.shapes)}")
        for i, s in enumerate(st.session_state.shapes):
            if s["type"] == "polygon":
                n      = len(s["points"])
                n_show = n - 1 if (n > 2 and s["points"][0] == s["points"][-1]) else n
                st.write(f"  {i+1}. Polygon — {n_show} vertices")
            else:
                st.write(f"  {i+1}. Circle — center {tuple(map(int, s['center']))}, r={int(s['radius'])}")
        if st.button("🗑️ Clear all shapes" if lang == "en" else "🗑️ 全図形をクリア"):
            st.session_state.shapes = []
            st.rerun()
    else:
        st.caption("No shapes committed yet." if lang == "en"
                   else "まだ図形がコミットされていません。")

    if st.button("📌 Commit Shape" if lang == "en" else "📌 図形をコミット"):
        data = st.session_state.pending_json
        if data and "objects" in data and len(data["objects"]) > 0:
            obj      = data["objects"][-1]
            obj_type = obj.get("type", "")
            if obj_type in ("polygon", "path"):
                abs_pts = parse_polygon_points(obj)
                if abs_pts and len(abs_pts) >= 3:
                    mapped = [[p[0] * sx, p[1] * sy] for p in abs_pts]
                    st.session_state.shapes.append({"type": "polygon", "points": mapped})
                    st.rerun()
                else:
                    st.warning("Polygon not closed. Right-click to close, then Commit." if lang == "en"
                               else "ポリゴンが閉じていません。右クリックで閉じてからコミットしてください。")
            elif obj_type == "circle":
                left_o  = float(obj.get("left",   0))
                top_o   = float(obj.get("top",    0))
                scale_x = float(obj.get("scaleX", 1.0))
                radius  = float(obj.get("radius", obj.get("width", 0) / 2))
                cx      = (left_o + radius) * sx
                cy      = (top_o  + radius) * sy
                r       = radius * scale_x * (sx + sy) / 2.0
                st.session_state.shapes.append({"type": "circle", "center": [cx, cy], "radius": r})
                st.rerun()
            else:
                st.warning(f"Unsupported shape type: {obj_type}")
        else:
            st.warning("No shape found to commit. Draw something first." if lang == "en"
                       else "コミットする図形が見つかりません。まず図形を描いてください。")

    st.markdown("---")
    if st.button("💾 Generate & Save Mask" if lang == "en" else "💾 マスクを生成して保存",
                 type="primary",
                 disabled=len(st.session_state.shapes) == 0):
        mask_uint8         = rasterize_shapes(H, W, st.session_state.shapes)
        mask_bool          = mask_uint8.astype(bool)
        save_mask(selected_folder_path, selected_stem, mask_bool)
        overlay            = img_np.copy()
        overlay[mask_bool] = (overlay[mask_bool] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
        st.image(overlay,
                 caption="Saved mask — green = foreground" if lang == "en"
                 else "保存済みマスク — 緑 = 前景",
                 use_container_width=True)
        st.success(f"Saved to masks/{selected_stem}_mask.npy ✅")
        st.session_state.shapes = []

    with st.expander("🎨 Drawing Settings" if lang == "en" else "🎨 描画設定", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.session_state.draw_mode    = st.radio(
                "Tool" if lang == "en" else "ツール",
                ["Polygon", "Circle"],
                index=0 if st.session_state.draw_mode == "Polygon" else 1
            )
            st.session_state.stroke_width = st.slider(
                "Outline width" if lang == "en" else "輪郭の幅",
                1, 10, st.session_state.stroke_width
            )
        with c2:
            st.session_state.stroke_color = st.color_picker(
                "Outline color" if lang == "en" else "輪郭の色",
                st.session_state.stroke_color
            )
            st.session_state.fill_color   = st.color_picker(
                "Fill color" if lang == "en" else "塗りつぶし色",
                st.session_state.fill_color
            )
        with c3:
            st.session_state.fill_alpha   = st.slider(
                "Fill opacity" if lang == "en" else "塗りつぶし透明度",
                0.0, 1.0, st.session_state.fill_alpha, 0.05
            )
