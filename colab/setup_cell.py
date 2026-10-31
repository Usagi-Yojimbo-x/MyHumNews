# ---- paste this into Colab AFTER drive.mount("/content/drive") ----
import os, glob, zipfile, pathlib

# Find xlmr_bundle.zip anywhere in your Drive (first match wins).
hits = glob.glob("/content/drive/MyDrive/**/xlmr_bundle.zip", recursive=True)
if not hits:
    raise SystemExit(
        "xlmr_bundle.zip not found in Drive.\n"
        "Upload it to drive.google.com first, then re-run this cell.\n"
        "Looked in: /content/drive/MyDrive/**/")
BUNDLE = hits[0]
print("found:", BUNDLE)

WORK = "/content/work"
os.makedirs(WORK, exist_ok=True)
with zipfile.ZipFile(BUNDLE) as z:
    z.extractall(WORK)
os.chdir(WORK)

# Checkpoint lives on Drive so a disconnect costs one fold, not five.
CKPT_DIR = "/content/drive/MyDrive/myhumnews"
os.makedirs(CKPT_DIR, exist_ok=True)
CKPT = f"{CKPT_DIR}/xlmr_ckpt.json"
os.environ["CKPT"] = CKPT

print("\ncwd:", os.getcwd())
for p in sorted(pathlib.Path(".").rglob("*")):
    if p.is_file():
        print(f"  {p}  {p.stat().st_size/1024:,.0f} KB")
print("\ncheckpoint ->", CKPT)
print("resuming from existing checkpoint" if os.path.exists(CKPT)
      else "fresh run (no checkpoint yet)")
