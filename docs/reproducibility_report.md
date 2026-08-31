# Reproducibility Report
Generated on: 2026-07-10 17:12:04

## Environment Details
- **Git Branch**: `main`
- **Git Commit**: `5bae08b051b21e42f616eda6a014eca269672290`
- **Python Version**: 3.14.3 (main, Feb  3 2026, 15:32:20) [Clang 17.0.0 (clang-1700.6.3.2)]
- **PyTorch Version**: 2.13.0
- **Compute Device**: MPS (Apple Silicon)

## Experiment Configuration
```yaml
# =============================================================================
# ADL Project Configuration
# Dual-Branch CNN for Non-Melanoma Dermoscopic Classification
# =============================================================================

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
seed: 42
deterministic: true  # torch.backends.cudnn.deterministic

# ---------------------------------------------------------------------------
# Device
# Options: auto | cuda | mps | cpu
# 'auto' selects cuda > mps > cpu automatically
# ---------------------------------------------------------------------------
device: auto

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
dataset:
  name: ham10000                    # ham10000 | isic2019
  root: data/raw/HAM10000           # Path to extracted dataset folder
  metadata_csv: HAM10000_metadata.csv
  image_dirs:
    - HAM10000_images_part_1
    - HAM10000_images_part_2
  mask_dir: null                    # Set to mask folder path if available (ISIC 2018)
  image_size: 224
  num_classes: 7
  class_names:
    - MEL
    - NV
    - BCC
    - AKIEC
    - BKL
    - DF
    - VASC
  # Class distribution in HAM10000 (used for focal loss alpha computation)
  class_counts:
    MEL: 1113
    NV: 6705
    BCC: 514
    AKIEC: 327
    BKL: 1099
    DF: 115
    VASC: 142

# ---------------------------------------------------------------------------
# Dataset Splits
# ---------------------------------------------------------------------------
split:
  train: 0.70
  val: 0.15
  test: 0.15
  stratify: true
  patient_aware: true               # Prevent patient-level data leakage

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
augmentation:
  random_resized_crop:
    height: 224
    width: 224
    scale: [0.8, 1.0]
  horizontal_flip_p: 0.5
  vertical_flip_p: 0.5
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
  elastic_transform_p: 0.3
  coarse_dropout_p: 0.2
  hair_augmentation: true           # Synthetic dermoscopic hair overlay
  # ImageNet normalization statistics
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
dataloader:
  batch_size: 32                    # Adjust: 32 for M4 MPS, 8 for RTX 3050
  num_workers: 4                    # Match to CPU core count
  pin_memory: false                 # Set true for CUDA, false for MPS/CPU
  use_weighted_sampler: true        # Oversample minority classes during training

# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
model:
  name: dual_branch                 # dual_branch | resnet50 | densenet201 | efficientnet
  # --- Dual-Branch Architecture ---
  dual_branch:
    shallow_wide:
      channels: [256, 512, 1024]    # Progressive channel expansion (texture pathway)
      pretrained_init: true         # Initialise from WideResNet-50-2 weights
    deep_narrow:
      base_channels: 64
      bottleneck_channels: 256
      num_blocks: [2, 2, 3, 3]      # Residual blocks per stage (structure pathway)
      pretrained_init: true         # Initialise from DenseNet-121 weights
    fusion:
      texture_dim: 1024
      structure_dim: 256
      hidden_dim: 512
      output_dim: 256
      dropout_1: 0.4
      dropout_2: 0.3
  # --- Multi-Task Learning settings ---
  mtl:
    enabled: false                  # Set true to use MTLDualBranchNet
    seg_lambda: 0.5                 # Weight for segmentation auxiliary loss
    cls_lambda: 1.0                 # Weight for classification loss

# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------
loss:
  type: focal                       # focal | cross_entropy
  focal:
    gamma: 2.0
    alpha: auto                     # 'auto' = compute from inverse class frequency
    reduction: mean
  label_smoothing: 0.1

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
optimizer:
  type: adamw
  lr: 1.0e-4
  weight_decay: 1.0e-4
  betas: [0.9, 0.999]

# ---------------------------------------------------------------------------
# Learning Rate Scheduler
# ---------------------------------------------------------------------------
scheduler:
  type: cosine_with_warmup
  warmup_epochs: 5
  min_lr: 1.0e-6
  T_max: 50                         # Total cosine annealing period (epochs)

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
training:
  epochs: 60
  grad_accum_steps: 1              # Increase to 2-4 for RTX 3050 (4 GB VRAM)
  mixed_precision: false           # true for CUDA; MPS has partial support only
  early_stopping_patience: 10
  save_dir: results/
  log_interval: 10                  # Log metrics every N batches

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
evaluation:
  metrics:
    - auc_macro
    - f1_macro
    - balanced_accuracy
    - recall_per_class
    - confusion_matrix
  save_predictions: true
  output_dir: results/eval/

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
paths:
  checkpoint_dir: checkpoints/
  log_dir: logs/
  results_dir: results/
  best_model: checkpoints/best_model.pth

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging:
  level: INFO                       # DEBUG | INFO | WARNING | ERROR | CRITICAL
  log_to_file: true
  log_filename: logs/train.log
  tensorboard: true
  tensorboard_dir: logs/tensorboard

# ---------------------------------------------------------------------------
# Explainability — Grad-CAM
# ---------------------------------------------------------------------------
gradcam:
  enabled: true
  shallow_target_layer: shallow_branch.block3.conv   # Last conv of shallow branch
  deep_target_layer: deep_branch.stage4.conv_layers  # Last conv of deep branch
  save_dir: results/gradcam/
  alpha: 0.4                        # Heatmap overlay opacity

# ---------------------------------------------------------------------------
# Uncertainty Quantification — Conformal Prediction
# ---------------------------------------------------------------------------
conformal:
  alpha: 0.1                        # Miscoverage rate → 90 % coverage guarantee
  cal_fraction: 0.2                 # Fraction of val set used for calibration
  method: raps                      # raps | lac (simple non-conformity score)
  raps:
    k_reg: 5
    lambda_reg: 0.01

# ---------------------------------------------------------------------------
# Uncertainty Quantification — MC Dropout
# ---------------------------------------------------------------------------
mc_dropout:
  n_samples: 50
  dropout_p: 0.3                    # Inference-time dropout probability

# ---------------------------------------------------------------------------
# Federated Learning (future phase, disabled by default)
# ---------------------------------------------------------------------------
federated:
  enabled: false
  num_clients: 5
  rounds: 20
  local_epochs: 5
  aggregation: fedavg
  differential_privacy:
    enabled: false
    noise_multiplier: 1.0
    max_grad_norm: 1.0

```

## Python Packages
<details>
<summary>Click to expand `pip freeze`</summary>

```text
albucore==0.0.24
albumentations==2.0.8
annotated-types==0.7.0
anyio==4.14.1
appnope==0.1.4
argon2-cffi==25.1.0
argon2-cffi-bindings==25.1.0
arrow==1.4.0
asttokens==3.0.1
async-lru==2.3.0
attrs==26.1.0
babel==2.18.0
beautifulsoup4==4.15.0
bleach==6.4.0
certifi==2026.6.17
cffi==2.1.0
charset-normalizer==3.4.9
click==8.4.2
comm==0.2.3
contourpy==1.3.3
coverage==7.15.0
cycler==0.12.1
debugpy==1.8.21
decorator==5.3.1
defusedxml==0.7.1
executing==2.2.1
fastjsonschema==2.21.2
filelock==3.29.7
fonttools==4.63.0
fqdn==1.5.1
fsspec==2026.6.0
h11==0.16.0
hf-xet==1.5.1
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.22.0
idna==3.18
iniconfig==2.3.0
ipykernel==7.3.0
ipython==9.15.0
ipython_pygments_lexers==1.1.1
ipywidgets==8.1.8
isoduration==20.11.0
jedi==0.20.0
Jinja2==3.1.6
joblib==1.5.3
json5==0.15.0
jsonpointer==3.1.1
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter==1.1.1
jupyter-console==6.6.3
jupyter-events==0.12.1
jupyter-lsp==2.3.1
jupyter_builder==1.0.2
jupyter_client==8.9.1
jupyter_core==5.9.1
jupyter_server==2.20.0
jupyter_server_terminals==0.5.4
jupyterlab==4.6.1
jupyterlab_pygments==0.3.0
jupyterlab_server==2.28.0
jupyterlab_widgets==3.0.16
kiwisolver==1.5.0
lark==1.3.1
lightning-utilities==0.15.3
markdown-it-py==4.2.0
MarkupSafe==3.0.3
matplotlib==3.11.0
matplotlib-inline==0.2.2
mdurl==0.1.2
mistune==3.3.3
mpmath==1.3.0
narwhals==2.23.0
nbclient==0.11.0
nbconvert==7.17.1
nbformat==5.10.4
nest-asyncio2==1.7.2
networkx==3.6.1
notebook==7.6.0
notebook_shim==0.2.4
numpy==2.5.1
opencv-python==5.0.0.93
opencv-python-headless==5.0.0.93
packaging==26.2
pandas==3.0.3
pandocfilters==1.5.1
parso==0.8.7
pexpect==4.9.0
pillow==12.3.0
platformdirs==4.10.0
pluggy==1.6.0
prometheus_client==0.25.0
prompt_toolkit==3.0.52
psutil==7.2.2
ptyprocess==0.7.0
pure_eval==0.2.3
pycparser==3.0
pydantic==2.13.4
pydantic_core==2.46.4
Pygments==2.20.0
pyparsing==3.3.2
pytest==9.1.1
pytest-cov==7.1.0
python-dateutil==2.9.0.post0
python-json-logger==4.1.0
PyYAML==6.0.3
pyzmq==27.1.0
referencing==0.37.0
requests==2.34.2
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rfc3987-syntax==1.1.0
rich==15.0.0
rpds-py==2026.6.3
safetensors==0.8.0
scikit-learn==1.9.0
scipy==1.18.0
seaborn==0.13.2
Send2Trash==2.1.0
setuptools==83.0.0
simsimd==6.5.16
six==1.17.0
soupsieve==2.8.4
stack-data==0.6.3
stringzilla==4.6.2
sympy==1.14.0
terminado==0.18.1
threadpoolctl==3.6.0
timm==1.0.27
tinycss2==1.5.1
torch==2.13.0
torchmetrics==1.9.0
torchvision==0.28.0
tornado==6.5.7
tqdm==4.68.4
traitlets==5.15.1
typing-inspection==0.4.2
typing_extensions==4.16.0
tzdata==2026.2
uri-template==1.3.0
urllib3==2.7.0
wcwidth==0.8.2
webcolors==25.10.0
webencodings==0.5.1
websocket-client==1.9.0
widgetsnbextension==4.0.15
```
</details>
