# -*- encoding: utf-8 -*-
'''
@File    :   direct_inference.py
@Time    :   2023/01/29 14:15:45
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   script for direct infernece of STU-Net, 
             output is formatted in JSON like nnUNet
'''

import argparse
import os
import os.path as osp
from tqdm import tqdm
import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, load_json
from nnunet.inference.predict import predict_cases
from nnunet.evaluation.evaluator import aggregate_scores
from label_sys import label_sys_dict, label_mapping, totalseg_cls2idx

def read_nifti(filename):
    """
    Read a NIfTI file and return the image object and data array.
    
    :param filename: Path to the NIfTI file
    :return: Tuple of (image object, data array)
    """
    img = nib.load(filename)
    arr = img.get_fdata()
    return img, arr

def save_nifti(arr, img, output_path):
    """
    Save a data array as a NIfTI file, using an original image object for metadata.
    
    :param arr: Modified data array
    :param img: Original image object
    :param output_path: Path to save the new NIfTI file
    """
    if arr.shape != img.shape:
        raise ValueError("The shape of the modified data array must match the original image shape.")
    
    new_img = nib.Nifti1Image(arr, img.affine, img.header)
    nib.save(new_img, output_path)

def map_all_data(test_ref_pairs, map_out_dir=None, label_info=None):
    os.makedirs(map_out_dir, exist_ok=True)
    print("mapping...")
    mapped_test_ref_pairs = []
    for test, ref in test_ref_pairs:
        # load test get arr
        test_img, test_arr = read_nifti(test)
        ref_img, ref_arr = read_nifti(ref)
        mapped_test_arr = np.zeros_like(test_arr, dtype=np.uint8)
        mapped_ref_arr = np.zeros_like(ref_arr, dtype=np.uint8)
        labels = dict()
        for idx, (local_values, local_cls_name) in enumerate(label_info["label_sys"].items()):
            global_cls_name = label_info["label_mapping"][local_cls_name]
            labels[idx+1] = local_cls_name

            if(global_cls_name.startswith("agg:")):
                global_values = [label_info["totalseg_cls2idx"][gcn] for gcn in global_cls_name[4:].split(",")]
            else:
                global_values = [label_info["totalseg_cls2idx"][global_cls_name]]
            # print("map", global_values, "to", local_cls_name, int(idx+1))
            for v in global_values:
                mapped_test_arr[test_arr==v] = int(idx+1)

            if(local_values.startswith("agg:")):
                local_values = [int(v) for v in local_values[4:].split(",")]
            else:
                local_values = [int(local_values)]
            # print("map", local_values, "to", local_cls_name, int(idx+1))
            for v in local_values:
                mapped_ref_arr[ref_arr==v] = int(idx+1)

        mapped_test_path = join(map_out_dir, "test_"+osp.basename(test))
        mapped_ref_path = join(map_out_dir, "ref_"+osp.basename(ref))
        mapped_test_ref_pairs.append((mapped_test_path, mapped_ref_path))
        save_nifti(mapped_test_arr, test_img, mapped_test_path)
        save_nifti(mapped_ref_arr, ref_img, mapped_ref_path)

    return mapped_test_ref_pairs, labels



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('trainer', type=str, help='Trainer name')
    parser.add_argument('test_dir', type=str, help='dir to test data, should contains `imagesTr`, `labelsTr` and `dataset.json`')
    parser.add_argument('output_dir', type=str, help='dir to save inference results for further metric computation')
    parser.add_argument('--mode', type=str, default='default', choices=['default', 'test', 'metric', 'split'], help='Mode of operation')
    parser.add_argument('--src_dataset', type=str, default='Task101_TotalSegmentator', help='Src Dataset name')
    parser.add_argument('--num_parts', type=int, default=1, help='Number of parts for split mode')
    parser.add_argument('--part_id', type=int, default=0, help='Part ID for split mode')

    args = parser.parse_args()

    args.target_dataset = osp.basename(args.test_dir)
    model = f"{args.trainer}__nnUNetPlansv2.1"
    val_fold = "all"
    test = args.mode == 'test'
    assert args.mode in ("default", "test", "metric", "split"), f"unknown mode '{args.mode}'"
    if args.mode == "split":
        part_id = args.part_id
        num_parts = args.num_parts
    else:
        part_id, num_parts = 0, 1
    results_folder = os.getenv('RESULTS_FOLDER')
    if results_folder is None:
        raise EnvironmentError("The environment variable 'RESULTS_FOLDER' is not set")

    parameter_folder = f'{results_folder}/nnUNet/3d_fullres/{args.src_dataset}'
    input_folder = f'{args.test_dir}/imagesTr'
    gt_folder = f'{args.test_dir}/labelsTr'
    pred_output_folder = f'{args.output_dir}/pred/{args.target_dataset}'
    json_output_folder = f'{args.output_dir}/json/{args.target_dataset}'
    mapped_data_folder = f'{args.output_dir}/mapped_pred/{args.target_dataset}'
    print(f"compute metrics of fold {val_fold}")

    test_files = subfiles(input_folder, suffix='.nii.gz', join=False)
    if (test): test_files = test_files[:2]

    input_files = [join(input_folder, tf) for tf in test_files]
    output_files = [join(pred_output_folder, model, tf) for tf in test_files]

    # handle data slicing
    if (num_parts > 1):
        input_files = input_files[part_id::num_parts]
        output_files = output_files[part_id::num_parts]
        print(f"split data into {num_parts} parts({part_id}), number:", len(input_files))

    # custom num_thread to avoid memory error
    num_threads = 4
    if (args.target_dataset in ["Task003_Liver"]):
        num_threads = 2

    # ----------- direct inference ------------------
    if (args.mode != "metric"):
        predict_cases(join(parameter_folder, model), [[i] for i in input_files], output_files, 
                      folds=0, save_npz=False, checkpoint_name=args.trainer.split("_")[-1]+"_ep4k",
                      num_threads_preprocessing=num_threads, num_threads_nifti_save=num_threads, 
                      segs_from_prev_stage=None, do_tta=False,
                      mixed_precision=True, overwrite_existing=False, all_in_gpu=False, step_size=0.5,)

    # ----------- compute metric --------------------
    print("start to compute metrics")
    json_output_dir = f"{json_output_folder}/{model}"
    if (test):
        json_output_dir = osp.join(osp.dirname(json_output_dir), "test_" + osp.basename(json_output_dir))
    os.makedirs(json_output_dir, exist_ok=True)
    data_list = output_files 
    label_sys = label_sys_dict[args.target_dataset]
    print("labels:", label_sys)
    pbar = tqdm(data_list)

    dataset_json_content = load_json(f"{args.test_dir}/dataset.json")
    image2label = dict()
    for image_label_pair in dataset_json_content["training"]:
        image2label[osp.basename(image_label_pair["image"])] = osp.basename(image_label_pair["label"])

    test_ref_pairs = []
    for data in pbar:
        image_key = osp.basename(data).replace("_0000.nii.gz", ".nii.gz") 
        label_fname = image2label[image_key]
        label_path = osp.join(gt_folder, label_fname)
        test_ref_pairs.append((data, label_path))
    print("all data", test_ref_pairs)

    mapped_test_ref_pairs, labels = map_all_data(
        test_ref_pairs,
        map_out_dir=join(mapped_data_folder, model),
        label_info={
            "label_sys": label_sys, 
            "label_mapping": label_mapping,
            "totalseg_cls2idx": totalseg_cls2idx,
        })

    aggregate_scores(
        mapped_test_ref_pairs, 
        json_output_file=join(json_output_dir, "summary.json"),
        num_threads=8, 
        labels=labels)
    print("finish computing metrics")
