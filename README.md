# STU-Net
Scalable and Transferable Medical Image Segmentation Models Enpowered by Large-scale Supervised Pre-training

[[`arxiv Paper`](https://arxiv.org/abs/2304.06716)]
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
## TotalSegmentator Trainer Models

| Model Name | Crop Size | #Params | FLOPs | Download Link |
|:------:|:-------:|:-----:|:---------:| :-------|
| STU-Net-S | 128x128x128 | 14.6M | 0.13T | [Baidu Netdisk](https://pan.baidu.com/s/1ZBfOhaTvjvhcgXKGNe_gWg?pwd=soz7) \| [Google Drive](https://drive.google.com/file/d/1HReH6dDrEuXgHPrsw7OrHSjvEUF3f4mv/view?usp=sharing)|
| STU-Net-B | 128x128x128 | 58.26M | 0.51T | [Baidu Netdisk](https://pan.baidu.com/s/1a17XmOGiGSgbEvK-acSOSg?pwd=91w3) \| [Google Drive](https://drive.google.com/file/d/1BHCp1Ort-OaVFwaZmvsG4qHiKiPeNb4h/view?usp=share_link)|
| STU-Net-L | 128x128x128 | 440.30M | 3.81T | [Baidu Netdisk](https://pan.baidu.com/s/1WOLoTrzCLYyJXZnITGK6jg?pwd=91pt) \| [Google Drive](https://drive.google.com/file/d/1KA1eXWWf_xAoJg5KHYrxTmfiz7wxGhHS/view?usp=share_link)|
| STU-Net-H | 128x128x128 | 1457.33M | 12.60T | [Baidu Netdisk](https://pan.baidu.com/s/1CinTvceZuvdEEWGcaJEuEA?pwd=bk9n) \| [Google Drive](https://drive.google.com/file/d/1Qrq7oGPJ7ileFHWOAxwpeWdaB6hySptU/view?usp=share_link)|

### Fine-tuning on downstream tasks
To perform fine-tuning on downstream tasks, use the following command with the base model as an example:
```
python run_finetuning.py 3d_fullres STUNetTrainer_base_ft TASKID FOLD -pretrained_weights MODEL
```
Please note that you may need to adjust the learning rate according to the specific downstream task. To do this, modify the learning rate in the corresponding Trainer (e.g., STUNetTrainer_base_ft) for the task.

# Citations
If you find this repository useful, please consider citing our paper:
```
@article{huang2023stu,
  title={STU-Net: Scalable and Transferable Medical Image Segmentation Models Empowered by Large-Scale Supervised Pre-training},
  author={Huang, Ziyan and Wang, Haoyu and Deng, Zhongying and Ye, Jin and Su, Yanzhou and Sun, Hui and He, Junjun and Gu, Yun and Gu, Lixu and Zhang, Shaoting and Qiao, Yu},
  journal={arXiv preprint arXiv:2304.06716},
  year={2023}
}
```
