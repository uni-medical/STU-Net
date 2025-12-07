#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Head & Neck Cancer Segmentation Inference Script
Using STU-Net Small trained on HECKTOR dataset

Model outputs:
    - Label 0: Background
    - Label 1: GTVp (Gross Tumor Volume - Primary tumor)
    - Label 2: GTVn (Gross Tumor Volume - Nodal metastases)

*** IMPORTANT: This model requires BOTH CT and PET scans! ***

Input requirements:
    - CT scan (channel 0) - named as *_0000.nii.gz
    - PET scan (channel 1) - named as *_0001.nii.gz
    - Both must be co-registered (same dimensions and orientation)
    
Usage:
    # From folder (files must have _0000 and _0001 suffixes):
    python run_hn_inference.py -i /path/to/images -o /path/to/output
    
    # From specific CT and PET files:
    python run_hn_inference.py --ct /path/to/CT.nii.gz --pet /path/to/PET.nii.gz -o /path/to/output
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add nnUNet-2.2 to path
SCRIPT_DIR = Path(__file__).parent.absolute()
NNUNET_PATH = SCRIPT_DIR / "nnUNet-2.2"
sys.path.insert(0, str(NNUNET_PATH))

from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def get_model_folder():
    """Get the path to the trained model folder"""
    return SCRIPT_DIR / "STUNetTrainer_small__nnUNetPlans__3d_fullres"


def run_inference(input_folder=None, output_folder=None, input_files=None, 
                  folds=(0,), use_gpu=True, save_probabilities=False):
    """
    Run Head & Neck cancer segmentation inference
    
    Args:
        input_folder: Folder containing input images (with _0000 and _0001 suffixes for CT and PET)
        output_folder: Folder to save predictions
        input_files: List of tuples [(ct_path, pet_path), ...] for direct file input
        folds: Tuple of fold numbers to use for ensembling (default: (0,))
        use_gpu: Whether to use GPU (default: True)
        save_probabilities: Whether to save probability maps (default: False)
    """
    model_folder = get_model_folder()
    
    # Check if model exists
    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder not found: {model_folder}")
    
    # Set device
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda', 0)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize predictor
    print("\nInitializing STU-Net predictor...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,  # Test-time augmentation
        perform_everything_on_gpu=use_gpu and torch.cuda.is_available(),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    # Load model
    print(f"Loading model from: {model_folder}")
    print(f"Using folds: {folds}")
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=folds,
        checkpoint_name='checkpoint_final.pth',
    )
    
    # Run prediction
    print(f"\nRunning inference...")
    print(f"Output folder: {output_folder}")
    
    if input_folder:
        # Predict from folder
        predictor.predict_from_files(
            input_folder,
            output_folder,
            save_probabilities=save_probabilities,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )
    elif input_files:
        # Predict from specific files
        # input_files should be list of lists: [[ct_0000.nii.gz, pet_0001.nii.gz], ...]
        output_files = []
        for files in input_files:
            ct_file = files[0]
            base_name = Path(ct_file).name.replace('_0000.nii.gz', '.nii.gz').replace('.nii.gz', '_pred.nii.gz')
            output_files.append(join(output_folder, base_name))
        
        predictor.predict_from_files(
            input_files,
            output_files,
            save_probabilities=save_probabilities,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE!")
    print("="*60)
    print(f"\nOutput saved to: {output_folder}")
    print("\nLabel mapping:")
    print("  0 = Background")
    print("  1 = GTVp (Primary Gross Tumor Volume)")
    print("  2 = GTVn (Nodal Gross Tumor Volume)")


def prepare_input_files(ct_path, pet_path, temp_folder):
    """
    Prepare input files with correct naming convention for nnUNet
    CT should be *_0000.nii.gz, PET should be *_0001.nii.gz
    """
    import shutil
    import nibabel as nib
    
    os.makedirs(temp_folder, exist_ok=True)
    
    # Get base name from CT file
    ct_name = Path(ct_path).stem.replace('.nii', '')
    
    # Copy/rename files
    ct_dest = join(temp_folder, f"{ct_name}_0000.nii.gz")
    pet_dest = join(temp_folder, f"{ct_name}_0001.nii.gz")
    
    # Handle both .nii and .nii.gz
    if ct_path.endswith('.nii.gz'):
        shutil.copy(ct_path, ct_dest)
    else:
        img = nib.load(ct_path)
        nib.save(img, ct_dest)
    
    if pet_path.endswith('.nii.gz'):
        shutil.copy(pet_path, pet_dest)
    else:
        img = nib.load(pet_path)
        nib.save(img, pet_dest)
    
    print(f"Prepared input files in: {temp_folder}")
    print(f"  CT:  {ct_dest}")
    print(f"  PET: {pet_dest}")
    
    return temp_folder


def main():
    parser = argparse.ArgumentParser(
        description='Head & Neck Cancer Segmentation using STU-Net',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict from folder (files must have _0000 and _0001 suffixes):
  python run_hn_inference.py -i input_data/CHUM-001 -o output_data/CHUM-001
  
  # Predict from specific CT and PET files:
  python run_hn_inference.py --ct CT.nii.gz --pet PET.nii.gz -o output_data/patient1
  
  # Use multiple folds for ensembling:
  python run_hn_inference.py -i input_data -o output_data --folds 0 1 2 3 4
  
  # Use CPU only:
  python run_hn_inference.py -i input_data -o output_data --cpu
        """
    )
    
    parser.add_argument('-i', '--input_folder', type=str, 
                        help='Input folder containing images (with _0000/_0001 suffixes)')
    parser.add_argument('-o', '--output_folder', type=str, required=True,
                        help='Output folder for predictions')
    parser.add_argument('--ct', type=str, help='Path to CT scan (will be prepared as _0000)')
    parser.add_argument('--pet', type=str, help='Path to PET scan (will be prepared as _0001)')
    parser.add_argument('--folds', type=int, nargs='+', default=[0],
                        help='Fold(s) to use for prediction (default: 0). Use multiple for ensembling.')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    parser.add_argument('--save_probs', action='store_true', help='Save probability maps')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input_folder is None and (args.ct is None or args.pet is None):
        parser.error("Either --input_folder OR both --ct and --pet must be provided")
    
    if args.ct and args.pet:
        # Prepare files and use temp folder as input
        temp_folder = join(args.output_folder, 'temp_input')
        input_folder = prepare_input_files(args.ct, args.pet, temp_folder)
    else:
        input_folder = args.input_folder
    
    # Run inference
    run_inference(
        input_folder=input_folder,
        output_folder=args.output_folder,
        folds=tuple(args.folds),
        use_gpu=not args.cpu,
        save_probabilities=args.save_probs
    )


if __name__ == '__main__':
    main()
