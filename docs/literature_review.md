# Literature Review Database

| Title | Authors | Year | Dataset | Architecture | Metrics (Acc/AUC) | Strengths | Weaknesses | How our approach differs |
|---|---|---|---|---|---|---|---|---|
| Deep residual learning for image recognition | He et al. | 2016 | ImageNet | ResNet | - | Very deep training | General purpose, not medical | We use a dual-branch custom CNN |
| HAM10000 dataset for classification | Tschandl et al. | 2018 | HAM10000 | - | - | Standardised dataset | Heavy class imbalance | Focal loss & conformal prediction |
