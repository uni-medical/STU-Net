# STU-Net
Scalable and Transferable Medical Image Segmentation Models Enpowered by Large-scale Supervised Pre-training
<p float="left">
  <img src="assets/fig_bubble.png?raw=true" width="47.5%" />
  <img src="assets/fig_model.png?raw=true" width="47.5%" /> 
</p>

# Environments and Requirements:
Our models are built based on [nnUNetV1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1). You should meet the requirements of nnUNet.
Copy the following files in this repo to your nnUNet repository.
```
copy /network_training/* nnunet/training/network_training/
copy /network_architecture/* nnunet/network_architecture/
copy run_finetuning.py nnunet/run/
```

# Pre-trained Models:
### TotalSegmentator trained models

| name | crop size | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:|
| STU-Net-S | 128x128x128 | 14.6M | 0.13T | Coming soon |
| STU-Net-B | 128x128x128 | 58.26M | 0.51T | Coming soon |
| STU-Net-L | 128x128x128 | 440.30M | 3.81T | Coming soon |
| STU-Net-H | 128x128x128 | 1457.33M | 12.60T | Coming soon |

