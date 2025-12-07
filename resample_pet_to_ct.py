#!/usr/bin/env python3
"""
Resample PET images to match CT resolution for HECKTOR dataset.
nnUNet requires all modalities to have the same dimensions.

What this script does:
1. CT images are typically 512x512 pixels (high resolution)
2. PET images are typically 128x128 pixels (lower resolution)
3. nnUNet requires all modalities to have SAME dimensions
4. This script resamples (upscales) PET to match CT resolution
5. Uses linear interpolation to preserve image quality
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt


def visualize_before_after(ct_path, pet_original_path, pet_resampled_path, output_dir):
    """Create visualization comparing CT, original PET, and resampled PET."""
    
    # Load images
    ct_img = nib.load(str(ct_path)).get_fdata()
    pet_orig = nib.load(str(pet_original_path)).get_fdata()
    pet_resampled = nib.load(str(pet_resampled_path)).get_fdata()
    
    # Get middle slice for visualization
    ct_slice = ct_img.shape[2] // 2
    pet_orig_slice = pet_orig.shape[2] // 2
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Axial views (top-down)
    axes[0, 0].imshow(ct_img[:, :, ct_slice].T, cmap='gray', origin='lower')
    axes[0, 0].set_title(f'CT (axial)\nShape: {ct_img.shape}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pet_orig[:, :, pet_orig_slice].T, cmap='hot', origin='lower')
    axes[0, 1].set_title(f'PET Original (axial)\nShape: {pet_orig.shape}')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pet_resampled[:, :, ct_slice].T, cmap='hot', origin='lower')
    axes[0, 2].set_title(f'PET Resampled (axial)\nShape: {pet_resampled.shape}')
    axes[0, 2].axis('off')
    
    # Row 2: Overlay and comparison
    # CT with PET overlay (before)
    axes[1, 0].imshow(ct_img[:, :, ct_slice].T, cmap='gray', origin='lower', alpha=1.0)
    axes[1, 0].set_title('CT only')
    axes[1, 0].axis('off')
    
    # Side-by-side PET comparison
    axes[1, 1].imshow(pet_orig[:, :, pet_orig_slice].T, cmap='hot', origin='lower')
    axes[1, 1].set_title(f'PET Original\n{pet_orig.shape[0]}x{pet_orig.shape[1]} pixels')
    axes[1, 1].axis('off')
    
    # CT with resampled PET overlay
    axes[1, 2].imshow(ct_img[:, :, ct_slice].T, cmap='gray', origin='lower', alpha=0.7)
    axes[1, 2].imshow(pet_resampled[:, :, ct_slice].T, cmap='hot', origin='lower', alpha=0.5)
    axes[1, 2].set_title('CT + Resampled PET Overlay')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'PET Resampling Visualization: {Path(ct_path).stem}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f"{Path(ct_path).stem}_resampling_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization: {output_path.name}")
    return output_path


def resample_pet_to_ct(ct_path, pet_path, output_pet_path):
    """Resample PET to match CT resolution and size."""
    
    # Read images with SimpleITK for better resampling
    ct_sitk = sitk.ReadImage(str(ct_path))
    pet_sitk = sitk.ReadImage(str(pet_path))
    
    # Create resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())
    
    # Resample PET to CT space
    pet_resampled = resampler.Execute(pet_sitk)
    
    # Save
    sitk.WriteImage(pet_resampled, str(output_pet_path))
    
    return pet_resampled

def main():
    base_dir = Path("/home/santhi/Documents/HNCancer/nnUNet_data/nnUNet_raw_data_base/nnUNet_raw_data/Dataset500_HECKTOR")
    
    # Create output directory for visualizations
    viz_dir = Path("/home/santhi/Documents/HNCancer/visualizations")
    viz_dir.mkdir(exist_ok=True)
    print(f"Saving visualizations to: {viz_dir}")
    
    # Process training images
    images_tr = base_dir / "imagesTr"
    
    # Get all CT files (modality 0000)
    ct_files = sorted(images_tr.glob("*_0000.nii.gz"))
    
    print(f"\n{'='*60}")
    print(f"PET RESAMPLING FOR HECKTOR DATASET")
    print(f"{'='*60}")
    print(f"\nWhy this is needed:")
    print(f"  - CT images: 512x512 resolution (high detail)")
    print(f"  - PET images: 128x128 resolution (lower detail)")
    print(f"  - nnUNet requires same dimensions for all modalities")
    print(f"  - Solution: Upsample PET to match CT using interpolation")
    print(f"\n{'='*60}")
    print(f"Found {len(ct_files)} CT files to process")
    
    for ct_path in ct_files:
        # Get corresponding PET file
        pet_path = str(ct_path).replace("_0000.nii.gz", "_0001.nii.gz")
        pet_path = Path(pet_path)
        
        if not pet_path.exists():
            print(f"  Warning: No PET found for {ct_path.name}")
            continue
        
        # Read original shapes
        ct_img = nib.load(str(ct_path))
        pet_img = nib.load(str(pet_path))
        
        print(f"\n{ct_path.stem}:")
        print(f"  CT shape:  {ct_img.shape} - Resolution: {ct_img.header.get_zooms()[:3]}")
        print(f"  PET shape: {pet_img.shape} - Resolution: {pet_img.header.get_zooms()[:3]}")
        
        if ct_img.shape != pet_img.shape:
            print(f"  ⚠️  Shapes MISMATCH - Resampling PET to match CT...")
            
            # Backup original PET with proper extension
            backup_path = str(pet_path).replace(".nii.gz", "_original.nii.gz")
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(str(pet_path), backup_path)
            
            # Resample and save
            resample_pet_to_ct(ct_path, backup_path, pet_path)
            
            # Verify
            pet_resampled = nib.load(str(pet_path))
            print(f"  ✓ PET shape (after): {pet_resampled.shape}")
            
            # Create visualization
            visualize_before_after(ct_path, backup_path, pet_path, viz_dir)
        else:
            print(f"  ✓ Shapes already match, skipping")
    
    # Process test images
    images_ts = base_dir / "imagesTs"
    if images_ts.exists():
        ct_files_ts = sorted(images_ts.glob("*_0000.nii.gz"))
        print(f"\n\n{'='*60}")
        print(f"Processing TEST images")
        print(f"{'='*60}")
        print(f"Found {len(ct_files_ts)} test CT files to process")
        
        for ct_path in ct_files_ts:
            pet_path = str(ct_path).replace("_0000.nii.gz", "_0001.nii.gz")
            pet_path = Path(pet_path)
            
            if not pet_path.exists():
                continue
            
            ct_img = nib.load(str(ct_path))
            pet_img = nib.load(str(pet_path))
            
            print(f"\n{ct_path.stem}:")
            print(f"  CT shape:  {ct_img.shape}")
            print(f"  PET shape: {pet_img.shape}")
            
            if ct_img.shape != pet_img.shape:
                print(f"  ⚠️  Shapes MISMATCH - Resampling PET to match CT...")
                
                # Backup original PET with proper extension
                backup_path = str(pet_path).replace(".nii.gz", "_original.nii.gz")
                if not os.path.exists(backup_path):
                    import shutil
                    shutil.copy2(str(pet_path), backup_path)
                
                resample_pet_to_ct(ct_path, backup_path, pet_path)
                
                pet_resampled = nib.load(str(pet_path))
                print(f"  ✓ PET shape (after): {pet_resampled.shape}")
                
                # Create visualization
                visualize_before_after(ct_path, backup_path, pet_path, viz_dir)
            else:
                print(f"  ✓ Shapes already match, skipping")
    
    print("\n" + "="*60)
    print("✓ PET RESAMPLING COMPLETE!")
    print("="*60)
    print(f"\nVisualizations saved to: {viz_dir}")
    print(f"You can now run: nnUNetv2_plan_and_preprocess -d 500")
    print("="*60)

if __name__ == "__main__":
    main()
