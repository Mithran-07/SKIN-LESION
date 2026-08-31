"""
Final verification script — Step 1: Checkpoint, model, inference.
"""
import io, sys, os, json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, "C:/ADL")
from models.baselines.efficientnet_baseline import EfficientNetBaseline

CK_PATH = "checkpoints/efficientnet_b4/best_checkpoint.pth"
RESULTS = {}

# ── 1. File exists and size ────────────────────────────────────────────────
assert os.path.exists(CK_PATH), "Checkpoint file missing!"
size_mb = os.path.getsize(CK_PATH) / (1024 * 1024)
print(f"[1] File exists: True")
print(f"[2] File size: {size_mb:.2f} MB")
RESULTS["checkpoint_exists"] = True
RESULTS["checkpoint_size_mb"] = round(size_mb, 2)

# ── 2. Load checkpoint ─────────────────────────────────────────────────────
ck = torch.load(CK_PATH, map_location="cpu", weights_only=False)
print(f"[3] Checkpoint keys: {list(ck.keys())}")
print(f"[4] Epoch: {ck.get('epoch')}")
RESULTS["epoch"] = ck.get("epoch")

# ── 3. Load model ──────────────────────────────────────────────────────────
model = EfficientNetBaseline(num_classes=7, pretrained=False)
state = ck["state_dict"]
state = {k.replace("module.", ""): v for k, v in state.items()}

# Detect bare backbone keys (no 'model.' prefix)
sample_key = next(iter(state))
model_keys = set(model.state_dict().keys())
if sample_key not in model_keys and ("model." + sample_key) in model_keys:
    print("[INFO] Adapting checkpoint: adding 'model.' prefix")
    state = {"model." + k: v for k, v in state.items()}

missing, unexpected = model.load_state_dict(state, strict=False)
print(f"[5] Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
assert len(missing) == 0, f"Missing keys: {missing}"
RESULTS["missing_keys"] = len(missing)
RESULTS["unexpected_keys"] = len(unexpected)

# ── 4. Run inference ───────────────────────────────────────────────────────
model.eval()
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    logits, _, _ = model(x)
probs = F.softmax(logits, dim=1).squeeze()
probs_list = probs.tolist()
print(f"[6] Output shape: {logits.shape}")
print(f"[7] Probs sum: {sum(probs_list):.6f}")
print(f"[8] Num classes: {logits.shape[1]}")
assert logits.shape[1] == 7, "Wrong num classes!"
assert abs(sum(probs_list) - 1.0) < 1e-4, "Probs don't sum to 1!"
RESULTS["num_classes"] = 7
RESULTS["probs_sum"] = round(sum(probs_list), 6)

# ── 5. Test with a real-ish image (random RGB) ─────────────────────────────
from torchvision import transforms
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
fake = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
tensor = tf(fake).unsqueeze(0)
with torch.no_grad():
    logits2, _, _ = model(tensor)
probs2 = F.softmax(logits2, dim=1).squeeze().tolist()
cls_names = ["akiec","bcc","bkl","df","mel","nv","vasc"]
pred = cls_names[int(np.argmax(probs2))]
print(f"[9] Inference on fake image: class={pred}, prob={max(probs2):.4f}")
RESULTS["sample_inference_class"] = pred
RESULTS["sample_inference_prob"] = round(max(probs2), 4)

print("\n[PASS] All checkpoint and inference checks passed.")
with open("artifacts/verify_checkpoint.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
print("[OK] Results saved to artifacts/verify_checkpoint.json")
