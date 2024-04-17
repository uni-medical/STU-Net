### Using Our Models for Inference
To use our trained models pretrained on [AbdomenAtlasMini](https://github.com/MrGiovanni/AbdomenAtlas) to conduct inference on CT images, please first organize the file structures in your `RESULTS_FOLDER/nnUNet/3d_fullres/` as follows. You can download the necessary files from our [Baidu Netdisk Download](https://pan.baidu.com/s/13-DG4Opn3IrxLehTRrs7sg?pwd=2t75) or [Google Drive link](https://drive.google.com/drive/folders/1mt5suOG9RkVg4lV85_RMnJ5dZTm_tRo2?usp=sharing):
```
- Task200_AbdomenAtlasMini/
  - STUNetTrainer_small_ep2k__nnUNetPlansv2.1/
    - plans.pkl
    - fold_all/
      - model_final_checkpoint.model
      - model_final_checkpoint.model.pkl
  - STUNetTrainer_base_ep2k__nnUNetPlansv2.1/
    - plans.pkl
    - fold_all/
      - model_final_checkpoint.model
      - model_final_checkpoint.model.pkl
  - STUNetTrainer_large_ep2k__nnUNetPlansv2.1/
    - plans.pkl
    - fold_all/
      - model_final_checkpoint.model
      - model_final_checkpoint.model.pkl
  - STUNetTrainer_huge_ep2k__nnUNetPlansv2.1/
    - plans.pkl
    - fold_all/
      - model_final_checkpoint.model
      - model_final_checkpoint.model.pkl
```
To conduct inference, you can use following command (base model for example):
```
nnUNet_predict -i INPUT_PATH -o OUTPUT_PATH -t 200 -m 3d_fullres -f all -tr STUNetTrainer_base_ep2k
```
For much faster inference speed with minimal performance loss, it is recommended to use the following command:
```
nnUNet_predict -i INPUT_PATH -o OUTPUT_PATH -t 200 -m 3d_fullres -f all -tr STUNetTrainer_base_ep2k --mode fast --disable_tta
```

The categories corresponding to the label values can be found in the [label_orders](label_orders.json) file within our repository