# segmentation_unet_deeplab_hrnet
Comparison of U-Net, DeepLabV3 and HRNet for semantic segmentation: brain stroke CT scans and ships in satellite imagery (PyTorch)
# Semantic Segmentation: U-Net vs DeepLabV3 vs HRNet

Comparison of three deep learning architectures for semantic segmentation on two different tasks: **brain stroke detection on CT scans** and **ship detection in satellite imagery**. Everything is implemented in PyTorch and trained on a Tesla T4 GPU (Kaggle).

## Overview

| Project | Domain | Models compared |
|---|---|---|
| Brain stroke segmentation | Medical imaging (CT) | U-Net, DeepLabV3, HRNet |
| Ship segmentation | Satellite imagery | U-Net, DeepLabV3 (ResNet50), HRNet |

## Project 1: Brain Stroke Segmentation (CT)

**Dataset:** Brain Stroke CT Dataset (Kaggle)
- 1,130 ischemia images, 1,093 hemorrhage images, 4,427 normal images
- Lesion masks were only available as colored overlays, so they were rebuilt automatically (Dice of 0.9995 against the true masks on 200 images)
- Final evaluation on an external set of 200 images never seen during training

**Training setup:** BCE + Dice loss, 40 epochs, same protocol for all models.

| Model | Parameters | Dice | IoU |
|---|---|---|---|
| U-Net | 7.8M | 0.800 | 0.778 |
| DeepLabV3 (ResNet50) | 39.6M | 0.845 | 0.818 |
| HRNet-W18 | 11.5M | 0.845 | 0.822 |

**Key findings**
- HRNet gives the best trade-off: same accuracy as DeepLabV3 with about 3.5x fewer parameters.
- It also leads on validation Dice (0.71 vs 0.65 for DeepLabV3 and 0.61 for U-Net).
- U-Net remains a strong lightweight baseline (about 0.80 Dice).
- Training cost differs a lot: roughly 1 min/epoch for U-Net, 2 min for DeepLabV3 and 5 min for HRNet.

## Project 2: Ship Segmentation in Satellite Imagery

**Dataset:** [Ships in Satellite Imagery](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery) (Kaggle)
- 4,000 thumbnails of 80x80 pixels (1,000 ship / 3,000 no-ship)
- 8 large satellite scenes without annotations
- Train / validation split: 3,400 / 600 images

**Method:** the dataset has no pixel-level masks, so uniform pseudo-masks are built from the image labels. Large scenes are then segmented with sliding-window inference to produce a dense probability map.

| Model | Parameters | Validation accuracy |
|---|---|---|
| U-Net (from scratch) | 7.8M | about 99.7% |
| DeepLabV3 (ResNet50, pretrained) | about 42M | about 99.8% |
| HRNet (compact, from scratch) | 0.42M | about 99.7% |

**Key findings**
- All models perform almost identically on this easier task.
- The compact HRNet matches the others with roughly 100x fewer parameters than DeepLabV3, so the lightest model is enough here.

## Main Takeaways

1. On a hard problem (small lesions, low contrast), the architecture really matters.
2. On a simpler problem, all models are equivalent and the lightest one is enough.
3. An external test set with real masks is essential for an honest evaluation.
4. Annotation quality matters as much as the choice of model.

## Limitations

- In the brain stroke project, DeepLabV3 was trained without pretrained weights, while HRNet was pretrained, so the comparison is not fully equal.
- In the ship project, scores are computed on uniform pseudo-masks. They mainly measure ship presence detection, not fine pixel-level segmentation.

## Repository Structure

```
brain-stroke/   notebooks for U-Net, DeepLabV3, HRNet
ships/          notebooks for U-Net, DeepLabV3, ResNet, HRNet
```

## How to Run

1. Open the notebooks on Kaggle (recommended) or locally with a GPU.
2. Add the corresponding dataset to the notebook.
3. Install the dependencies:

```
pip install torch torchvision numpy pandas opencv-python matplotlib
```

## Next Steps

- Compare all models under identical pretraining conditions
- Multi-class segmentation (ischemia vs hemorrhage)
- Real pixel-level annotations for the ship dataset

## Author

