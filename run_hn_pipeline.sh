#!/bin/bash
# =============================================================
# Head & Neck Cancer Segmentation + Visualization Pipeline
# =============================================================
# 
# Usage:
#   ./run_hn_pipeline.sh <PATIENT_ID> <CT_PATH> <PET_PATH>
#
# Example:
#   ./run_hn_pipeline.sh CHUM-001 /path/to/ct.nii.gz /path/to/pet.nii.gz
#
# Output Structure (all in one folder):
#   output_data/PATIENT_ID/
#   ├── input/
#   │   ├── PATIENT_ID_0000.nii.gz  (CT)
#   │   └── PATIENT_ID_0001.nii.gz  (PET)
#   ├── segmentation/
#   │   └── PATIENT_ID.nii.gz       (Tumor segmentation)
#   ├── anatomy/
#   │   └── anatomy.nii             (TotalSegmentator output)
#   └── visualization/
#       ├── PATIENT_ID_3d_with_anatomy.html
#       ├── PATIENT_ID_anatomy_multiview.png
#       └── *.vtk meshes
# =============================================================

set -e  # Exit on error

# Check arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <PATIENT_DIR> [CT_PATH] [PET_PATH]"
    echo ""
    echo "Examples:"
    echo "  # Auto-detect CT/PET from HECKTOR format directory:"
    echo "  $0 /path/to/data/CHUM-002"
    echo ""
    echo "  # Specify CT and PET files explicitly:"
    echo "  $0 CHUM-001 /path/to/ct.nii.gz /path/to/pet.nii.gz"
    exit 1
fi

# Check if first argument is a directory (auto-detect mode) or patient ID
if [ -d "$1" ]; then
    # Auto-detect mode: find CT and PET in the directory
    DATA_DIR="$1"
    PATIENT_ID=$(basename "$DATA_DIR")
    
    echo "Auto-detecting CT and PET files in: ${DATA_DIR}"
    
    # Look for CT file (patterns: *__CT.nii.gz, *_CT.nii.gz, *_0000.nii.gz)
    CT_PATH=$(find "$DATA_DIR" -maxdepth 1 -name "*__CT.nii.gz" -o -name "*_CT.nii.gz" -o -name "*_0000.nii.gz" 2>/dev/null | head -1)
    
    # Look for PET file (patterns: *__PT.nii.gz, *_PT.nii.gz, *_PET.nii.gz, *_0001.nii.gz)
    PET_PATH=$(find "$DATA_DIR" -maxdepth 1 -name "*__PT.nii.gz" -o -name "*_PT.nii.gz" -o -name "*_PET.nii.gz" -o -name "*_0001.nii.gz" 2>/dev/null | head -1)
    
    if [ -z "$CT_PATH" ]; then
        echo "ERROR: Could not find CT file in $DATA_DIR"
        echo "  Expected patterns: *__CT.nii.gz, *_CT.nii.gz, *_0000.nii.gz"
        exit 1
    fi
    
    if [ -z "$PET_PATH" ]; then
        echo "ERROR: Could not find PET file in $DATA_DIR"
        echo "  Expected patterns: *__PT.nii.gz, *_PT.nii.gz, *_PET.nii.gz, *_0001.nii.gz"
        exit 1
    fi
    
    echo "  Found CT:  ${CT_PATH}"
    echo "  Found PET: ${PET_PATH}"
else
    # Explicit mode: patient ID and file paths provided
    if [ "$#" -lt 3 ]; then
        echo "ERROR: When providing patient ID, CT and PET paths are required"
        echo "Usage: $0 <PATIENT_ID> <CT_PATH> <PET_PATH>"
        exit 1
    fi
    PATIENT_ID="$1"
    CT_PATH="$2"
    PET_PATH="$3"
fi

# Base directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}"

# Output directories - all in one patient folder
PATIENT_DIR="${BASE_DIR}/output_data/${PATIENT_ID}"
INPUT_DIR="${PATIENT_DIR}/input"
SEG_DIR="${PATIENT_DIR}/segmentation"
ANATOMY_DIR="${PATIENT_DIR}/anatomy"
VIZ_DIR="${PATIENT_DIR}/visualization"

echo "============================================================="
echo "H&N Cancer Segmentation Pipeline"
echo "============================================================="
echo "Patient ID: ${PATIENT_ID}"
echo "CT Path:    ${CT_PATH}"
echo "PET Path:   ${PET_PATH}"
echo "Output:     ${PATIENT_DIR}"
echo "============================================================="

# Validate input files exist
if [ ! -f "$CT_PATH" ]; then
    echo "ERROR: CT file not found: $CT_PATH"
    exit 1
fi

if [ ! -f "$PET_PATH" ]; then
    echo "ERROR: PET file not found: $PET_PATH"
    exit 1
fi

# Create directory structure
echo ""
echo "[1/5] Creating directory structure..."
mkdir -p "${INPUT_DIR}"
mkdir -p "${SEG_DIR}"
mkdir -p "${ANATOMY_DIR}"
mkdir -p "${VIZ_DIR}"

# Copy input files with nnUNet naming convention
echo ""
echo "[2/5] Preparing input files..."
cp "${CT_PATH}" "${INPUT_DIR}/${PATIENT_ID}_0000.nii.gz"
cp "${PET_PATH}" "${INPUT_DIR}/${PATIENT_ID}_0001.nii.gz"
echo "  ✓ CT  -> ${INPUT_DIR}/${PATIENT_ID}_0000.nii.gz"
echo "  ✓ PET -> ${INPUT_DIR}/${PATIENT_ID}_0001.nii.gz"

# Run TotalSegmentator for anatomical structures
echo ""
echo "[3/5] Running TotalSegmentator for anatomy..."
TotalSegmentator \
    -i "${INPUT_DIR}/${PATIENT_ID}_0000.nii.gz" \
    -o "${ANATOMY_DIR}/anatomy.nii" \
    --ml --fast

echo "  ✓ Anatomy saved to ${ANATOMY_DIR}/anatomy.nii"

# Run nnUNet tumor segmentation
echo ""
echo "[4/5] Running nnUNet tumor segmentation (GTVp + GTVn)..."
cd "${BASE_DIR}"

nnUNetv2_predict \
    -i "${INPUT_DIR}" \
    -o "${SEG_DIR}" \
    -d 001 \
    -c 3d_fullres \
    -tr STUNetTrainer_small \
    -f 9 \
    -p nnUNetPlans

echo "  ✓ Segmentation saved to ${SEG_DIR}/${PATIENT_ID}.nii.gz"

# Generate visualization
echo ""
echo "[5/5] Generating 3D visualization..."
python "${BASE_DIR}/visualize_hn_with_anatomy.py" \
    -i "${SEG_DIR}/${PATIENT_ID}.nii.gz" \
    -o "${VIZ_DIR}" \
    --anatomy "${ANATOMY_DIR}/anatomy.nii" \
    --patient-id "${PATIENT_ID}"

# Summary
echo ""
echo "============================================================="
echo "PIPELINE COMPLETE!"
echo "============================================================="
echo ""
echo "All outputs saved to: ${PATIENT_DIR}/"
echo ""
echo "Directory structure:"
echo "  ${PATIENT_DIR}/"
echo "  ├── input/"
echo "  │   ├── ${PATIENT_ID}_0000.nii.gz  (CT)"
echo "  │   └── ${PATIENT_ID}_0001.nii.gz  (PET)"
echo "  ├── segmentation/"
echo "  │   └── ${PATIENT_ID}.nii.gz       (Tumor: GTVp + GTVn)"
echo "  ├── anatomy/"
echo "  │   └── anatomy.nii                (TotalSegmentator)"
echo "  └── visualization/"
echo "      ├── ${PATIENT_ID}_3d_with_anatomy.html"
echo "      ├── ${PATIENT_ID}_anatomy_multiview.png"
echo "      └── *.vtk meshes"
echo ""
echo "To view the 3D visualization:"
echo "  xdg-open ${VIZ_DIR}/${PATIENT_ID}_3d_with_anatomy.html"
echo ""
