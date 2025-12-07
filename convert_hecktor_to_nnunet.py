#!/usr/bin/env python3
"""
Convert HECKTOR dataset to nnUNet format
"""
import os
import shutil
import json
from pathlib import Path

# Paths
source_dir = "/home/santhi/Documents/HNCancer/data"
target_base = "/home/santhi/Documents/HNCancer/nnUNet_data/nnUNet_raw_data_base/nnUNet_raw_data/Task500_HECKTOR"

# Get all patient folders
patient_folders = sorted([f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))])
print(f"Found {len(patient_folders)} patients: {patient_folders}")

# Split into training (9) and testing (2)
train_patients = patient_folders[:9]
test_patients = patient_folders[9:]

print(f"\nTraining patients ({len(train_patients)}): {train_patients}")
print(f"Testing patients ({len(test_patients)}): {test_patients}")

# Process training data
print("\n=== Processing Training Data ===")
for idx, patient in enumerate(train_patients):
    patient_dir = os.path.join(source_dir, patient)
    case_id = f"{idx:03d}"  # 000, 001, 002, etc.
    
    # Copy CT image (modality 0000)
    ct_src = os.path.join(patient_dir, f"{patient}__CT.nii.gz")
    ct_dst = os.path.join(target_base, "imagesTr", f"HECKTOR_{case_id}_0000.nii.gz")
    if os.path.exists(ct_src):
        shutil.copy2(ct_src, ct_dst)
        print(f"✓ Copied CT: {patient} → HECKTOR_{case_id}_0000.nii.gz")
    
    # Copy PET image (modality 0001)
    pt_src = os.path.join(patient_dir, f"{patient}__PT.nii.gz")
    pt_dst = os.path.join(target_base, "imagesTr", f"HECKTOR_{case_id}_0001.nii.gz")
    if os.path.exists(pt_src):
        shutil.copy2(pt_src, pt_dst)
        print(f"✓ Copied PET: {patient} → HECKTOR_{case_id}_0001.nii.gz")
    
    # Copy label
    label_src = os.path.join(patient_dir, f"{patient}.nii.gz")
    label_dst = os.path.join(target_base, "labelsTr", f"HECKTOR_{case_id}.nii.gz")
    if os.path.exists(label_src):
        shutil.copy2(label_src, label_dst)
        print(f"✓ Copied Label: {patient} → HECKTOR_{case_id}.nii.gz")

# Process testing data
print("\n=== Processing Testing Data ===")
for idx, patient in enumerate(test_patients):
    patient_dir = os.path.join(source_dir, patient)
    case_id = f"{idx:03d}"  # 000, 001 for test set
    
    # Copy CT image (modality 0000)
    ct_src = os.path.join(patient_dir, f"{patient}__CT.nii.gz")
    ct_dst = os.path.join(target_base, "imagesTs", f"HECKTOR_{case_id}_0000.nii.gz")
    if os.path.exists(ct_src):
        shutil.copy2(ct_src, ct_dst)
        print(f"✓ Copied CT: {patient} → HECKTOR_{case_id}_0000.nii.gz")
    
    # Copy PET image (modality 0001)
    pt_src = os.path.join(patient_dir, f"{patient}__PT.nii.gz")
    pt_dst = os.path.join(target_base, "imagesTs", f"HECKTOR_{case_id}_0001.nii.gz")
    if os.path.exists(pt_src):
        shutil.copy2(pt_src, pt_dst)
        print(f"✓ Copied PET: {patient} → HECKTOR_{case_id}_0001.nii.gz")

# Create dataset.json
print("\n=== Creating dataset.json ===")
dataset_json = {
    "name": "HECKTOR",
    "description": "Head and Neck Cancer Tumor Segmentation (HECKTOR 2025)",
    "tensorImageSize": "4D",
    "reference": "HECKTOR Challenge",
    "licence": "see HECKTOR challenge",
    "release": "1.0",
    "modality": {
        "0": "CT",
        "1": "PET"
    },
    "labels": {
        "0": "background",
        "1": "GTVp",
        "2": "GTVn"
    },
    "numTraining": len(train_patients),
    "numTest": len(test_patients),
    "training": [
        {
            "image": f"./imagesTr/HECKTOR_{i:03d}.nii.gz",
            "label": f"./labelsTr/HECKTOR_{i:03d}.nii.gz"
        }
        for i in range(len(train_patients))
    ],
    "test": [
        f"./imagesTs/HECKTOR_{i:03d}.nii.gz"
        for i in range(len(test_patients))
    ]
}

dataset_json_path = os.path.join(target_base, "dataset.json")
with open(dataset_json_path, 'w') as f:
    json.dump(dataset_json, f, indent=4)
print(f"✓ Created {dataset_json_path}")

print("\n" + "="*50)
print("✓ CONVERSION COMPLETE!")
print("="*50)
print(f"\nDataset location: {target_base}")
print(f"Training cases: {len(train_patients)}")
print(f"Testing cases: {len(test_patients)}")
print("\nNext steps:")
print("1. Run: nnUNet_plan_and_preprocess -t 500 --verify_dataset_integrity")
print("2. Fine-tune: python nnunetv2/run/run_finetuning_stunet.py Dataset500_HECKTOR 3d_fullres 0 -pretrained_weights MODEL -tr STUNetTrainer_base_ft")
