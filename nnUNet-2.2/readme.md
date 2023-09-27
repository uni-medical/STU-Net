# nnUNet Modifications

We have made specific changes to the nnUNet version 1.7 codebase. All other original nnUNet code remains unchanged and is only copied over.

## Changes

- Added `STUNet.py` in `nnunet/network_architecture/` for our network structure.
- Introduced `STUNetTrainer.py` in `nnunet/training/network_training/` for Trainer configurations.
- Included `run_finetuning.py` in `nnunet/run/` for fine-tuning with pre-trained weights.

