from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
import torch
from PIL import Image
import pandas as pd
import numpy as np
from itertools import compress
from collections import OrderedDict


def transforms_validation(image):
    crop_size = 224
    resize_size = 256
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    interpolation = InterpolationMode.BILINEAR
    transforms_val = transforms.Compose([
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=mean, std=std)
    ])
    image = Image.fromarray(np.uint8(image))
    image = transforms_val(image).reshape((1, 3, 224, 224))
    return image


def evaluate(model, image, cmnDf, class_txt_path):
    """
    Args:
        model:          loaded RegNet model 
        image:          numpy array (H x W x 3)
        cmnDf:          pd.DataFrame loaded from classes.csv
                        expected columns: 'Scientific Name', 'Common Name',
                                          'Order', 'Family', 'Role in Ecosystem'
        class_txt_path: path to classes.txt  (one scientific name per line,
                        index must match model output logit index)

    Returns:
        sciPred, cmnPred, order, family, role, confirmed, other_classes
    """
    model.eval()
    device = torch.device('cpu')
    image = transforms_validation(image)

    # Load class index → scientific name mapping from classes.txt
    with open(class_txt_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    with torch.inference_mode():
        image = image.to(device, non_blocking=True)
        output = model(image)

        # ── Energy-based out-of-distribution detection ──────────────────────
        T = 1
        energy = -(T * torch.logsumexp(output / T, dim=1)).item()
        confirmed = False
        other_classes = {}

        if energy < 11.49:
            confirmed = True
            smx = torch.nn.functional.softmax(output, dim=1).numpy()[0]

            # Collect all classes above the confidence threshold
            threshold = 1 - 0.935176
            op = smx >= threshold
            softmax_vals = list(compress(smx, op))
            names = list(compress(classes, op))

            # Sort descending by softmax score
            sorted_pairs = sorted(zip(softmax_vals, names), reverse=True)
            sortedSoftmax = [x for x, _ in sorted_pairs]
            sortedNames   = [x for _, x in sorted_pairs]

            if len(sortedSoftmax) > 1:
                sortedSoftmax = np.array(sortedSoftmax) / sum(sortedSoftmax) * 100
                softmax_class_dict = OrderedDict(zip(sortedNames, sortedSoftmax))
                for k in list(softmax_class_dict.keys())[1:]:
                    sc_name  = k
                    cmn_name = cmnDf.loc[
                        cmnDf['Scientific Name'] == sc_name, 'Common Name'
                    ].iloc[0]
                    other_classes[sc_name] = cmn_name

        # ── Top prediction ───────────────────────────────────────────────────
        smx    = torch.nn.functional.softmax(output, dim=1)
        op_ix  = torch.argmax(smx).item()
        sciPred = classes[op_ix]

        row     = cmnDf.loc[cmnDf['Scientific Name'] == sciPred]
        cmnPred = row['Common Name'].iloc[0]
        order   = row['Order'].iloc[0]
        family  = row['Family'].iloc[0]
        role    = row['Role in Ecosystem'].iloc[0]

    return sciPred, cmnPred, order, family, role, confirmed, other_classes
