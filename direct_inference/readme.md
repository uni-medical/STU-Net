# Direct Inference with STU-Net

This guide provides instructions on how to perform direct inference on external datasets using STU-Net.

## Prerequisites

Before proceeding with the direct inference, ensure you have completed the following steps:

1. **Environment Setup**: Follow the instructions in the [Get Started](https://github.com/blueyo0/STU-Net#get-started) section of the STU-Net repository to set up the necessary environment.

2. **Load Pre-trained Weights**: Download and prepare the pre-trained weights as described in the [Using Our Models for Inference](https://github.com/blueyo0/STU-Net#using-our-models-for-inference) section.

## Usage

Once the prerequisites are met, you can perform direct inference using the `direct_inference.py` script. The general syntax for the script is as follows:

```
python direct_inference.py <trainer> <test_dir> <output_dir>
```

- `<trainer>`: Specifies the trainer to be used.
- `<test_dir>`: The directory containing the external dataset. This directory should include subdirectories `imagesTr`, `labelsTr`, and a `dataset.json` file.
- `<output_dir>`: The directory where the output will be saved.

### Example

Here is an example of how to use the script:

```
python direct_inference.py STUNetTrainer_small example/Task032_AMOS22_Task1 results
```

### Output

After running the script, you can find the `summary.json` file, similar to the nnUNet validation output, at the following path:

```
<output_dir>/json/<dataset>/<trainer>
```

This file contains a summary of the inference results.

## Inference on a New Dataset

For datasets not included in the paper, you need to manually modify the label information in `label_sys.py` to ensure the correct mapping of labels and calculation of metric results. Follow these steps:

1. Refer to dictionaries like `AMOS_sys`, and add `<your local label dict>`. The format should be: `<index of local label> to <global label name>` (e.g. *"1":"liver"* or *"agg:1:2":"kidney"*, using 'agg:' to declare combinations of multiple labels).
2. In `label_sys_dict`, add a mapping from `<dataset name>` to `<your local label dict>`. The `<dataset name>` should correspond to your `<test_dir>` when inference.
3. **(Optional)** If the global label names in your `<your local label dict>` correspond to combinations of multiple labels in Totalsegmentator label system, add your mapping in `label_mapping` like *"kidney":"agg:kidney_left,kidney_right"*.

Then you can infer on your own dataset.
