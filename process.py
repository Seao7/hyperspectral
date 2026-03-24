import os
import json
import shutil
import numpy as np
from hsd_utils import read_HSD_from_file, HSD_to_RGB_save

# ── CONFIG ────────────────────────────────────────────────────────────────────
ANALYSIS_SPACE = "."
REGISTRY_FILE  = "name_registry.json"
HSD_EXT        = ".hsd"
RGB_CONFIG     = {"R_band": 55, "G_band": 35, "B_band": 23, "use_range": 3, "gamma": 2.2}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_registry():
    if not os.path.exists(REGISTRY_FILE):
        raise FileNotFoundError(f"{REGISTRY_FILE} not found — run setup.py first.")
    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def find_data_folders(root):
    folders = []
    for folder in os.listdir(root):
        path = os.path.join(root, folder)
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, "raw")):
            folders.append(path)
    return sorted(folders)

def clear_output_folders(processed_dir, rgb_dir, mean_spectrums_dir):
    """Wipe all outputs so every run starts completely fresh."""
    for folder in [processed_dir, rgb_dir, mean_spectrums_dir]:
        shutil.rmtree(folder)
        os.makedirs(folder)
    print("  🗑️   Cleared processed/, rgb_composites/, mean_spectrums/")

def load_references(references_folder):
    """
    0 files → (None, None)     → normalise only
    1 file  → (white, None)    → raw / white
    2 files → (white, dark)    → (raw - dark) / (white - dark)
    2+ files → brightest/darkest pair used, warning printed
    """
    hsd_files = sorted([f for f in os.listdir(references_folder) if f.endswith(HSD_EXT)])

    if len(hsd_files) == 0:
        return None, None

    if len(hsd_files) == 1:
        path = os.path.join(references_folder, hsd_files[0])
        print(f"  📄  Single reference: {hsd_files[0]}")
        print(f"  📐  Correction mode : raw / white")
        data, _, _, _ = read_HSD_from_file(path)
        return data.astype(np.float32), None

    if len(hsd_files) >= 2:
        if len(hsd_files) > 2:
            print(f"  ⚠️   {len(hsd_files)} files in references/ — using brightest as white, darkest as dark.")

        loaded = []
        for f in hsd_files:
            data, _, _, _ = read_HSD_from_file(os.path.join(references_folder, f))
            loaded.append((data.astype(np.float32), f))
        loaded.sort(key=lambda x: x[0].mean())

        dark,  dark_name  = loaded[0]
        white, white_name = loaded[-1]

        print(f"  📄  White reference : {white_name}")
        print(f"  📄  Dark reference  : {dark_name}")
        print(f"  📐  Correction mode : (raw - dark) / (white - dark)")
        return white, dark

def apply_correction(raw, white, dark=None):
    if dark is not None:
        denom = white - dark
        denom[denom == 0] = 1e-6
        corrected = (raw.astype(np.float32) - dark) / denom
    else:
        w = white.copy()
        w[w == 0] = 1e-6
        corrected = raw.astype(np.float32) / w
    return np.clip(corrected, 0, 1)

def get_scan_files(raw_dir):
    return sorted([f for f in os.listdir(raw_dir) if f.endswith(HSD_EXT)])

def get_display_name(stem, registry):
    entry = registry.get(stem, {})
    en    = entry.get("display_en", stem)
    ja    = entry.get("display_ja", "")
    return f"{en}  /  {ja}" if ja and ja != en else en

# ── FOLDER PROCESSOR ──────────────────────────────────────────────────────────
def process_folder(folder_path, registry):
    folder_name        = os.path.basename(folder_path)
    raw_dir            = os.path.join(folder_path, "raw")
    ref_dir            = os.path.join(folder_path, "references")
    processed_dir      = os.path.join(folder_path, "processed")
    rgb_dir            = os.path.join(folder_path, "rgb_composites")
    mean_spectrums_dir = os.path.join(folder_path, "mean_spectrums")

    print(f"\n▶  {folder_name}")

    for d in [raw_dir, ref_dir, processed_dir, rgb_dir, mean_spectrums_dir]:
        if not os.path.isdir(d):
            print(f"  ❌  Missing folder: {os.path.basename(d)} — run setup.py first.")
            return

    # ── Wipe outputs for a clean run ──────────────────────────────────────────
    clear_output_folders(processed_dir, rgb_dir, mean_spectrums_dir)

    # ── Load references ───────────────────────────────────────────────────────
    white_data, dark_data = load_references(ref_dir)
    if white_data is None:
        print("  ⚠️   No references found — normalising only (/ 65535)")

    # ── Get scan files ────────────────────────────────────────────────────────
    scan_files = get_scan_files(raw_dir)
    if not scan_files:
        print("  ⚠️   No scan files found in raw/")
        return

    print(f"  🔎  Found {len(scan_files)} scan(s)\n")

    # ── Process each scan ─────────────────────────────────────────────────────
    for filename in scan_files:
        stem         = os.path.splitext(filename)[0]
        display_name = get_display_name(stem, registry)
        raw_path     = os.path.join(raw_dir, filename)
        npy_path     = os.path.join(processed_dir, stem + ".npy")
        spectra_path = os.path.join(mean_spectrums_dir, stem + "_spectra.npy")
        rgb_out      = os.path.join(rgb_dir, stem)

        print(f"  ⏳  {display_name} ...", end=" ", flush=True)

        try:
            raw_data, _, _, _ = read_HSD_from_file(raw_path)

            if white_data is not None:
                corrected = apply_correction(raw_data, white_data, dark_data)
            else:
                corrected = raw_data.astype(np.float32) / 65535.0

            # Full corrected array → processed/
            np.save(npy_path, corrected)

            # Mean spectrum → mean_spectrums/
            np.save(spectra_path, corrected.mean(axis=(0, 1)))

            # RGB composite → rgb_composites/
            corrected_uint8 = (corrected * 255).astype(np.uint8)
            HSD_to_RGB_save(corrected_uint8, file_name=rgb_out, **RGB_CONFIG)

            print("✅")

        except Exception as e:
            print(f"❌  Error: {e}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔄  Starting processing...\n")

    registry     = load_registry()
    data_folders = find_data_folders(ANALYSIS_SPACE)

    if not data_folders:
        print("❌  No valid data folders found. Check that setup.py was run.")
        exit(1)

    for folder in data_folders:
        process_folder(folder, registry)

    print("\n✅  Processing complete. Run app.py to start the viewer.\n")
