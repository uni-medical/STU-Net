#!/bin/bash
# Head & Neck Cancer Segmentation Pipeline
# Uses STU-Net trained on HECKTOR dataset
#
# Usage:
#   ./run_pipeline.sh [OUTPUT_NAME]
#
# Arguments:
#   OUTPUT_NAME  Optional name for output subfolder (default: PATIENT_ID)
#
# IMPORTANT: Run this from the 'hncancer' conda environment
# Example: conda activate hncancer && ./run_pipeline.sh
# Example with custom output: conda activate hncancer && ./run_pipeline.sh CHUM-002_fold9
#
# Dependencies (in hncancer env):
#   - dicom2nifti==2.4.8  (required for TotalSegmentator with pydicom 2.4.x)
#   - TotalSegmentator
#   - nnunetv2
#   - pyvista

set -e  # Exit on error

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

# Configuration
PROJECT_DIR="$(pwd)"
DATA_DIR="$PROJECT_DIR/data"
INPUT_DIR="$PROJECT_DIR/input_data"
OUTPUT_DIR="$PROJECT_DIR/output_data"

# Patient configuration (modify these for your data)
PATIENT_ID="CHUM-002"
FOLDS="1"  # Use "0 1 2 3 4" for 5-fold ensemble (more accurate but slower)

# Output folder name (from argument or default to PATIENT_ID)
OUTPUT_NAME="${1:-$PATIENT_ID}"

# Device settings
USE_CPU="false"  # Set to "true" to force CPU inference

echo "$(timestamp) ================================================"
echo "$(timestamp)   HNCancer - Head & Neck Segmentation Pipeline"
echo "$(timestamp) ================================================"
echo "$(timestamp) "
echo "$(timestamp) Environment: $CONDA_DEFAULT_ENV"
echo "$(timestamp) Pipeline parameters:"
echo "$(timestamp)   PROJECT_DIR:  $PROJECT_DIR"
echo "$(timestamp)   DATA_DIR:     $DATA_DIR"
echo "$(timestamp)   PATIENT_ID:   $PATIENT_ID"
echo "$(timestamp)   OUTPUT_NAME:  $OUTPUT_NAME"
echo "$(timestamp)   FOLDS:        $FOLDS"
echo "$(timestamp)   USE_CPU:      $USE_CPU"
echo "$(timestamp) "

# Step 1: Prepare input data (resample PET to match CT)
echo "$(timestamp) STEP 1: Preparing input data..."

PATIENT_INPUT="$INPUT_DIR/${PATIENT_ID}"
PATIENT_OUTPUT="$OUTPUT_DIR/${OUTPUT_NAME}"
mkdir -p "$PATIENT_INPUT"
mkdir -p "$PATIENT_OUTPUT"

# Copy and rename CT and PET files
CT_FILE="$DATA_DIR/${PATIENT_ID}/${PATIENT_ID}__CT.nii.gz"
PET_FILE="$DATA_DIR/${PATIENT_ID}/${PATIENT_ID}__PT.nii.gz"

if [ ! -f "$CT_FILE" ]; then
    echo "$(timestamp) ERROR: CT file not found: $CT_FILE"
    exit 1
fi

if [ ! -f "$PET_FILE" ]; then
    echo "$(timestamp) ERROR: PET file not found: $PET_FILE"
    exit 1
fi

# Copy with nnUNet naming convention
cp "$CT_FILE" "$PATIENT_INPUT/${PATIENT_ID}_0000.nii.gz"
echo "$(timestamp)   CT copied: ${PATIENT_ID}_0000.nii.gz"

# Resample PET to match CT dimensions
echo "$(timestamp)   Resampling PET to match CT..."
python -c "
import SimpleITK as sitk

ct_path = '$PATIENT_INPUT/${PATIENT_ID}_0000.nii.gz'
pet_path = '$PET_FILE'
output_path = '$PATIENT_INPUT/${PATIENT_ID}_0001.nii.gz'

ct_sitk = sitk.ReadImage(ct_path)
pet_sitk = sitk.ReadImage(pet_path)

print(f'    CT size: {ct_sitk.GetSize()}')
print(f'    PET size: {pet_sitk.GetSize()}')

if ct_sitk.GetSize() != pet_sitk.GetSize():
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())
    pet_resampled = resampler.Execute(pet_sitk)
    sitk.WriteImage(pet_resampled, output_path)
    print(f'    PET resampled to: {pet_resampled.GetSize()}')
else:
    sitk.WriteImage(pet_sitk, output_path)
    print('    PET dimensions already match CT')
"
echo "$(timestamp) ✓ Step 1 complete"
echo "$(timestamp) "

# Step 2: Run TotalSegmentator for anatomical structures (optional)
echo "$(timestamp) STEP 2: Running TotalSegmentator for anatomy..."
ANATOMY_DIR="$PATIENT_OUTPUT/anatomy"
mkdir -p "$ANATOMY_DIR"

# Try to run TotalSegmentator, skip if it fails
ANATOMY_FILE="$ANATOMY_DIR/anatomy.nii"
if TotalSegmentator \
    -i "$PATIENT_INPUT/${PATIENT_ID}_0000.nii.gz" \
    -o "$ANATOMY_FILE" \
    --ml --fast 2>/dev/null; then
    echo "$(timestamp)   Anatomy saved to: $ANATOMY_FILE"
    echo "$(timestamp) ✓ Step 2 complete"
else
    echo "$(timestamp)   WARNING: TotalSegmentator failed or not available. Skipping anatomy."
    ANATOMY_FILE=""
    echo "$(timestamp) ✓ Step 2 skipped"
fi
echo "$(timestamp) "

# Step 3: Run STU-Net Inference
echo "$(timestamp) STEP 3: Running Head & Neck segmentation (STU-Net)..."

CPU_FLAG=""
if [ "$USE_CPU" = "true" ]; then
    CPU_FLAG="--cpu"
fi

python run_hn_inference.py \
    -i "$PATIENT_INPUT" \
    -o "$PATIENT_OUTPUT" \
    --folds $FOLDS \
    $CPU_FLAG

echo "$(timestamp) ✓ Step 3 complete"
echo "$(timestamp) "

# Step 4: 3D Visualization with Anatomy
echo "$(timestamp) STEP 4: Creating 3D visualization with anatomy..."

if [ -n "$ANATOMY_FILE" ] && [ -f "$ANATOMY_FILE" ]; then
    python visualize_hn_with_anatomy.py \
        -i "$PATIENT_OUTPUT/${PATIENT_ID}.nii.gz" \
        -o "$PATIENT_OUTPUT/visualization" \
        --anatomy "$ANATOMY_FILE" \
        --patient-id "$OUTPUT_NAME"
else
    # Fall back to basic visualization without anatomy
    python visualize_hn_segmentation.py \
        -i "$PATIENT_OUTPUT/${PATIENT_ID}.nii.gz" \
        -o "$PATIENT_OUTPUT/visualization" \
        --patient-id "$OUTPUT_NAME"
fi

echo "$(timestamp) ✓ Step 4 complete"
echo "$(timestamp) "

# Step 5: Compare with ground truth (if available)
GT_FILE="$DATA_DIR/${PATIENT_ID}/${PATIENT_ID}.nii.gz"
if [ -f "$GT_FILE" ]; then
    echo "$(timestamp) STEP 4: Comparing with ground truth..."
    python -c "
import nibabel as nib
import numpy as np

pred = nib.load('$PATIENT_OUTPUT/${PATIENT_ID}.nii.gz')
gt = nib.load('$GT_FILE')

pred_data = pred.get_fdata()
gt_data = gt.get_fdata()

print('  Results:')
for label, name in [(1, 'GTVp (Primary)'), (2, 'GTVn (Nodal)')]:
    pred_mask = (pred_data == label)
    gt_mask = (gt_data == label)
    
    if gt_mask.sum() > 0:
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum())
        print(f'    {name}: Dice = {dice:.3f}')
    else:
        print(f'    {name}: Not present in ground truth')
"
    echo "$(timestamp) ✓ Step 5 complete"
else
    echo "$(timestamp) STEP 5: Skipped (no ground truth available)"
fi

echo "$(timestamp) "
echo "$(timestamp) ================================================"
echo "$(timestamp)   Pipeline Complete!"
echo "$(timestamp) ================================================"
echo "$(timestamp) "
echo "$(timestamp) Output files:"
echo "$(timestamp)   Segmentation: $PATIENT_OUTPUT/${PATIENT_ID}.nii.gz"
echo "$(timestamp)   Visualization: $PATIENT_OUTPUT/visualization/"
echo "$(timestamp) "
echo "$(timestamp) Labels:"
echo "$(timestamp)   1 = GTVp (Primary Gross Tumor Volume)"
echo "$(timestamp)   2 = GTVn (Nodal Gross Tumor Volume)"
echo "$(timestamp) "