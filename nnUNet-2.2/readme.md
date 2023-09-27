# nnUNet Modifications

We have made specific changes to the nnUNet version 2.2 codebase. All other original nnUNet code remains unchanged and is only copied over.

## Changes

- Introduced `STUNetTrainer.py` in `nnunetv2/training/nnUNetTrainer/` for Trainer configurations.
- Included `run_finetuning_stunet.py` in `nnunetv2/run/` for fine-tuning with pre-trained weights.

