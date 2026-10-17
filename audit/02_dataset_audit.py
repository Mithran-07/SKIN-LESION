"""
AUDIT 02 — Dataset integrity + split leakage audit (read-only).
- Metadata integrity, image file existence/readability, dimensions
- Exact duplicate files (SHA-1 of bytes) and near-duplicates (64-bit dHash, Hamming <= 4)
- Split manifest verification: lesion overlap, image overlap, duplicate overlap across splits
- Re-runs data/splitter.patient_stratified_split(seed=42) IN MEMORY and diffs against splits/*.csv
Writes only to audit/outputs/.
"""
import hashlib, json, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HAM = Path("/Users/mithran/Documents/My projects/ADL MAIN/ADL.  LOQ/datasets/HAM10000")
OUT = ROOT / "audit" / "outputs"; OUT.mkdir(exist_ok=True)
R = {}

# ── Metadata
meta = pd.read_csv(HAM / "HAM10000_metadata.csv")
R["metadata_rows"] = len(meta)
R["metadata_columns"] = list(meta.columns)
R["unique_image_id"] = int(meta.image_id.nunique())
R["unique_lesion_id"] = int(meta.lesion_id.nunique())
R["missing_values_per_column"] = {c: int(v) for c, v in meta.isna().sum().items()}
R["sex_unknown"] = int((meta.sex == "unknown").sum())
R["dx_values"] = meta.dx.value_counts().to_dict()
R["dx_type_values"] = meta.dx_type.value_counts().to_dict()
imgs_per_lesion = meta.groupby("lesion_id").size()
R["images_per_lesion_distribution"] = {int(k): int(v) for k, v in imgs_per_lesion.value_counts().sort_index().items()}
R["lesions_with_multiple_dx"] = int((meta.groupby("lesion_id").dx.nunique() > 1).sum())
R["class_counts"] = meta.dx.value_counts().to_dict()
R["class_pct"] = (meta.dx.value_counts(normalize=True) * 100).round(2).to_dict()

# ── Image files
files = {p.stem: p for d in ["HAM10000_images_part_1", "HAM10000_images_part_2"] for p in (HAM / d).glob("*.jpg")}
R["image_files_found"] = len(files)
R["metadata_ids_without_file"] = sorted(set(meta.image_id) - set(files))
R["files_without_metadata"] = sorted(set(files) - set(meta.image_id))

sha, dh, dims, modes, bad = {}, {}, Counter(), Counter(), []
def dhash(img, n=8):
    g = np.asarray(img.convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)

for i, (iid, p) in enumerate(sorted(files.items())):
    b = p.read_bytes()
    sha[iid] = hashlib.sha1(b).hexdigest()
    try:
        with Image.open(p) as im:
            im.load()
            dims[im.size] += 1; modes[im.mode] += 1
            dh[iid] = dhash(im)
    except Exception as e:
        bad.append((iid, str(e)))
    if i % 2000 == 0: print(f"  scanned {i}/{len(files)}")
R["unreadable_images"] = bad
R["image_dimensions"] = {f"{w}x{h}": c for (w, h), c in dims.items()}
R["image_modes"] = dict(modes)

# ── Exact duplicates
by_sha = {}
for k, v in sha.items(): by_sha.setdefault(v, []).append(k)
exact_groups = [g for g in by_sha.values() if len(g) > 1]
R["exact_duplicate_groups"] = len(exact_groups)
R["exact_duplicate_examples"] = exact_groups[:10]

# ── Near duplicates (dHash Hamming <= 4)
ids = sorted(dh); H = np.array([dh[i] for i in ids], dtype=np.uint64)
pairs = []
for i in range(len(H)):
    x = np.bitwise_xor(H[i + 1:], H[i])
    # popcount
    d = np.zeros(len(x), dtype=np.int32); y = x.copy()
    for _ in range(64):
        d += (y & np.uint64(1)).astype(np.int32); y >>= np.uint64(1)
        if not y.any(): break
    for j in np.nonzero(d <= 4)[0]:
        pairs.append((ids[i], ids[i + 1 + j], int(d[j])))
R["near_duplicate_pairs_hamming_le4"] = len(pairs)

# ── Split manifests
sp = {s: pd.read_csv(ROOT / "splits" / f"{s}.csv") for s in ["train", "val", "test"]}
R["split_sizes"] = {s: len(d) for s, d in sp.items()}
R["split_lesions"] = {s: int(d.lesion_id.nunique()) for s, d in sp.items()}
R["split_class_counts"] = {s: d.dx.value_counts().reindex(["akiec","bcc","bkl","df","mel","nv","vasc"]).fillna(0).astype(int).to_dict() for s, d in sp.items()}
R["split_class_pct"] = {s: (d.dx.value_counts(normalize=True)*100).round(2).reindex(["akiec","bcc","bkl","df","mel","nv","vasc"]).to_dict() for s, d in sp.items()}
L = {s: set(d.lesion_id) for s, d in sp.items()}; I = {s: set(d.image_id) for s, d in sp.items()}
R["lesion_overlap"] = {"train_val": len(L["train"] & L["val"]), "train_test": len(L["train"] & L["test"]), "val_test": len(L["val"] & L["test"])}
R["image_overlap"] = {"train_val": len(I["train"] & I["val"]), "train_test": len(I["train"] & I["test"]), "val_test": len(I["val"] & I["test"])}
R["all_metadata_images_assigned"] = len(I["train"] | I["val"] | I["test"]) == len(meta)
split_of = {iid: s for s, ids_ in I.items() for iid in ids_}
lab_ok = all((sp[s].dx.map({"akiec":0,"bcc":1,"bkl":2,"df":3,"mel":4,"nv":5,"vasc":6}) == sp[s].label).all() for s in sp)
R["label_column_matches_HAM_LABEL_MAP"] = bool(lab_ok)
md = meta.set_index("image_id").dx
R["split_dx_matches_metadata"] = bool(all((sp[s].set_index("image_id").dx == md.reindex(sp[s].image_id)).all() for s in sp))

cross_exact = [g for g in exact_groups if len({split_of.get(i) for i in g}) > 1]
R["exact_duplicates_crossing_splits"] = len(cross_exact)
lesion_of = meta.set_index("image_id").lesion_id
cross_near = [(a, b, d) for a, b, d in pairs if split_of.get(a) != split_of.get(b)]
cross_near_diff_lesion = [(a, b, d) for a, b, d in cross_near if lesion_of.get(a) and lesion_of.get(b) and lesion_of.get(a) != lesion_of.get(b)]
R["near_duplicates_crossing_splits"] = len(cross_near)
R["near_duplicates_crossing_splits_examples"] = cross_near[:15]
R["near_dup_pairs_same_lesion"] = sum(1 for a, b, _ in pairs if lesion_of.get(a) and lesion_of.get(b) and lesion_of.get(a) == lesion_of.get(b))
R["near_dup_pairs_different_lesion"] = sum(1 for a, b, _ in pairs if lesion_of.get(a) and lesion_of.get(b) and lesion_of.get(a) != lesion_of.get(b))

# ── Stratification check: does GroupShuffleSplit actually stratify? Re-run in memory.
from data.splitter import patient_stratified_split
import data.splitter as S
S.HAM10000_DIR = HAM  # only used for img_path lookup
tr, va, te = patient_stratified_split(meta.copy(), seed=42)
R["regenerated_split_sizes"] = {"train": len(tr), "val": len(va), "test": len(te)}
R["regenerated_matches_manifest"] = {
    "train": set(tr.image_id) == I["train"], "val": set(va.image_id) == I["val"], "test": set(te.image_id) == I["test"]}

(OUT / "02_dataset.json").write_text(json.dumps(R, indent=2, default=str))
for k, v in R.items():
    s = json.dumps(v, default=str)
    print(f"{k}: {s[:400]}")
