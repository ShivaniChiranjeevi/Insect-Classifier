# Insect-Classifier

**Requirements:** Python ≥ 3.8  
Install PyTorch + torchvision from https://pytorch.org/get-started/previous-versions/

Additional dependencies:
```
pip install pandas numpy opencv-python
```

Model weights are available on [Zenodo](https://zenodo.org/records/14538000).

Download the full repository (which should contain):
```
your_classifier_dir/
├── model_github.pth   ← model weights
├── classes.txt        ← one scientific name per line (index = logit index)
└── classes.csv        ← metadata: Scientific Name, Common Name, Order, Family, Role in Ecosystem
```

---

## Quick-start example

```python
import torch
import torchvision
import cv2
import pandas as pd
from evaluate import evaluate   # evaluate.py from this repo

PATH_TO_CLASSIFIER = '/path/to/your_classifier_dir'
PATH_TO_IMAGE      = '/path/to/your_image.jpg'

# ── 1. Load model ──────────────────────────────────────────────────────────
weights = torch.load(
    PATH_TO_CLASSIFIER + '/model_github.pth',
    map_location=torch.device('cpu'),
    weights_only=False
)['model']                          # <-- note the ['model'] key

model = torchvision.models.regnet_y_32gf()
model.fc = torch.nn.Linear(3712, 2526)
model.load_state_dict(weights, strict=True)
torch.backends.cudnn.benchmark    = False
torch.backends.cudnn.deterministic = True
model.eval()

# ── 2. Load metadata ───────────────────────────────────────────────────────
cmnDf          = pd.read_csv(PATH_TO_CLASSIFIER + '/classes.csv')
class_txt_path = PATH_TO_CLASSIFIER + '/classes.txt'
# classes.csv must have columns:
#   'Scientific Name', 'Common Name', 'Order', 'Family', 'Role in Ecosystem'
# classes.txt must have one scientific name per line whose line index matches
#   the corresponding model output logit index.

# ── 3. Load & preprocess image ─────────────────────────────────────────────
image = cv2.imread(PATH_TO_IMAGE)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ── 4. Run inference ───────────────────────────────────────────────────────
sci, cmn, order, family, role, confirmed, other = evaluate(
    model, image, cmnDf, class_txt_path
)

print("Scientific Name:", sci)
print("Common Name:    ", cmn)
print("Order:          ", order)
print("Family:         ", family)
print("Role:           ", role)
print("Confirmed:      ", confirmed)
print("Other plausible:", other)
```

### Example output
```
Scientific Name:  Oxythyrea funesta
Common Name:      white spotted rose beetle
Order:            Coleoptera
Family:           Scarabaeidae
Role:             Pest
Confirmed:        True
Other plausible:  {}
```

---

## What changed vs. the old evaluate.py

| Issue | Old code | Fixed code |
|---|---|---|
| Class index source | `df['genus'] + ' ' + df['species']` from CSV | `classes.txt` (one name per line) — **critical** |
| OOD detection | Softmax threshold ≥ 0.97 | Energy-based threshold (energy < 11.49) |
| Weight loading | `torch.load(...)` directly | `torch.load(...)['model']` |
| Metadata lookup | Inline string concat | Lookup against `'Scientific Name'` column in CSV |

---

## License

Model weights are released under the  
**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.  
Permitted for non-commercial research with proper attribution.  
Full text: https://creativecommons.org/licenses/by-nc/4.0/
