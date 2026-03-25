import os
import json
import shutil
from deep_translator import GoogleTranslator

# ── CONFIG ──────────────────────────────────────────────────────────────────
ANALYSIS_SPACE = "."   # run setup.py from inside JA八女_土壌分析/
REGISTRY_FILE  = "name_registry.json"
SUBFOLDERS = ["raw", "references", "processed", "rgb_composites", "mean_spectrums", "masks"]
HSD_EXT        = ".hsd"

# ── STEP 1: Scan for data folders ───────────────────────────────────────────
def find_data_folders(root):
    """Find all folders directly under root that contain .hsd files."""
    found = []
    for folder in os.listdir(root):
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        has_hsd = any(f.endswith(HSD_EXT) for f in os.listdir(folder_path))
        if has_hsd:
            if ".ignore" not in folder_path:
                found.append(folder_path)
    return found

# ── STEP 2: Create folder structure ─────────────────────────────────────────
def create_structure(data_folder):
    for sub in SUBFOLDERS:
        path = os.path.join(data_folder, sub)
        os.makedirs(path, exist_ok=True)
    print(f"  📁 Created subfolders in: {os.path.basename(data_folder)}")

# ── STEP 3: Move .hsd files into raw/ ───────────────────────────────────────
def move_to_raw(data_folder):
    raw_path = os.path.join(data_folder, "raw")
    moved, skipped = [], []

    for f in os.listdir(data_folder):
        if f.endswith(HSD_EXT):
            src = os.path.join(data_folder, f)
            dst = os.path.join(raw_path, f)
            if os.path.exists(dst):
                skipped.append(f)
            else:
                shutil.move(src, dst)
                moved.append(f)

    print(f"  📦 Moved {len(moved)} files to raw/  |  Skipped {len(skipped)} (already exist)")
    return [os.path.splitext(f)[0] for f in moved + skipped]  # return stems

# ── STEP 4: Build / update name registry ────────────────────────────────────
def extract_suffix(filename, common_prefix):
    """Strip the common prefix to get the meaningful part of the name."""
    if filename.startswith(common_prefix):
        return filename[len(common_prefix):].strip("_")
    return filename

def find_common_prefix(names):
    if not names:
        return ""
    prefix = names[0]
    for name in names[1:]:
        while not name.startswith(prefix):
            prefix = prefix[:-1]
        if not prefix:
            break
    return prefix

def update_registry(all_stems):
    # Load existing registry
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {}

    # Find common prefix across all filenames to strip it
    common_prefix = find_common_prefix(all_stems)
    translator   = GoogleTranslator(source='ja', target='en')

    new_count = 0
    for stem in all_stems:
        if stem in registry:
            continue  # ← never overwrite existing entries

        suffix = extract_suffix(stem, common_prefix)
        try:
            translated = translator.translate(suffix)
        except Exception:
            translated = suffix  # fallback: use original if translation fails

        display_en = translated.title() if translated else suffix
        registry[stem] = {
            "ja":         stem,
            "en":         translated.lower().replace(" ", "_") if translated else suffix,
            "display_en": display_en,
            "display_ja": suffix
        }
        new_count += 1

    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

    print(f"  📖 Registry: {new_count} new entries added  |  {len(registry)} total entries")
    return registry

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔍 Scanning Analysis Space...\n")
    data_folders = find_data_folders(ANALYSIS_SPACE)

    if not data_folders:
        print("❌ No folders with .hsd files found. Are you running this from the right directory?")
        exit(1)

    all_stems = []
    for folder in data_folders:
        print(f"▶ Processing: {os.path.basename(folder)}")
        create_structure(folder)
        stems = move_to_raw(folder)
        all_stems.extend(stems)

    print(f"\n🌐 Building name registry...")
    update_registry(all_stems)

    print("\n✅ Setup complete. Your structure is ready.")
    print("   Next step: add dark/white references to each references/ folder, then run process.py\n")
