import os
import nibabel as nib
import numpy as np
import subprocess
import shutil
import tempfile

label_map = {
    1: "aorta",
    2: "gallbladder",
    3: "IVC",
    4: "left_kidney",
    5: "liver",
    6: "pancreas",
    7: "right_kidney",
    8: "spleen",
    9: "stomach",
}


def run_nnUNet_predict(input_folder, output_folder, trainer):
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    command = [
        "nnUNet_predict",
        "-i", input_folder,
        "-o", output_folder,
        "-t", "200",  
        "-m", "3d_fullres",  
        "-tr", trainer,
        "-f", "all"
    ]
    subprocess.run(command, check=True)


def convert_to_nnUNet_format(source_dir, nnUNet_input_dir):
    os.makedirs(nnUNet_input_dir, exist_ok=True)
    for case_name in os.listdir(source_dir):
        ct_file_path = os.path.join(source_dir, case_name, 'ct.nii.gz')
        if not os.path.isfile(ct_file_path):
            continue
        new_file_path = os.path.join(nnUNet_input_dir,case_name+'_0000.nii.gz')
        shutil.copyfile(ct_file_path, new_file_path)


def convert_prediction_to_AbdomenAtlas(nnUNet_output_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(nnUNet_output_dir):
        if not file.endswith('.nii.gz'):
            continue
        os.makedirs(os.path.join(output_dir, file[:-7], 'predictions'), exist_ok=True)
        file_path = os.path.join(nnUNet_output_dir, file)
        img = nib.load(file_path)
        img_data = img.get_fdata()
        for label, organ_name in label_map.items():
            organ_data = np.where(img_data == label, 1, 0)
            organ_img = nib.Nifti1Image(organ_data, img.affine, img.header)
            nib.save(organ_img, os.path.join(output_dir, file[:-7], 'predictions', organ_name+'.nii.gz'))

trainer = "STUNetTrainer_base_ep2k"
source_dir = '/mnt/d/tmp/AbdomenAtlas_test/test_examples/AbdomenAtlasTest/'
output_dir = '/mnt/d/tmp/AbdomenAtlas_test/test_examples/AbdomenAtlasPredict'

with tempfile.TemporaryDirectory() as nnUNet_input_dir, tempfile.TemporaryDirectory() as predictions_dir:
    convert_to_nnUNet_format(source_dir, nnUNet_input_dir)
    run_nnUNet_predict(nnUNet_input_dir, predictions_dir, trainer)
    convert_prediction_to_AbdomenAtlas(predictions_dir, output_dir)
