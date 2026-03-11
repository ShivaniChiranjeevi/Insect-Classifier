# Insect-Classifier
Packages and versions requirements:  Python 3.8 (minimum),  torch and torchvision can be downloaded at https://pytorch.org/get-started/previous-versions/


This tutorial guides you on how to load the pretrained insect classifier weights and use it for evaluation. The weights for the classifier are available [here](https://zenodo.org/records/14538000?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImRmYTc1MDk0LTJlNzYtNDlkYy1iNDZhLWI0NmU4Mzc3OWVhZiIsImRhdGEiOnt9LCJyYW5kb20iOiI2OGQxMDU4OWU2NDgxMjhiNGUxMTFhMDU4YTdiZTBkNiJ9.E3b6rnplZkUWSjta0whBI3_r8y1jixMa5JatyAcAPfJPX-NXaaqV-7ckeEQfOpDFvkQ7XoDHIyWvCUVkED-rng). Download the repository and navigate into its directory.
## Loading libraries

```{r test-python, engine='python'}
import torch
import torchvision
import evaluate
import cv2
```

## Creating the classifier model

```{r test-python, engine='python'}
model=torchvision.models.regnet_y_32gf()
```

## Loading weights of the model

```{r test-python, engine='python'}
weights=torch.load(PATH_TO_PTH_FILE+'/model.pth',map_location=torch.device('cpu'))
model.fc=torch.nn.Linear(3712,2526)
model.load_state_dict(weights,strict=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
model.eval()
```

## Function to load the image and predict Insect class

```{r test-python, engine='python'}
image=cv2.imread(PATH_TO_IMAGE)
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
result=evaluate.evaluate(model,image)
print(result)          
```

## License

The model weights are released under the  
Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

This allows researchers to use, modify, and redistribute the model for
non-commercial purposes with proper attribution.

Full license text: https://creativecommons.org/licenses/by-nc/4.0/
