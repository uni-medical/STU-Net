# -*- encoding: utf-8 -*-
'''
@File    :   direct_inference.py
@Time    :   2023/01/29 14:16:44
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   dict to map local label of dataset
             into our global label system
'''

'''
to use this script, add your custom local label_sys 
then, add the mapping into `label_sys_dict`

example:
Task003_Liver_sys = { 
   "agg:1,2": "liver", 
} # map local 1 & 2 to global category: `liver`
label_sys_dict = {
    "Task003_Liver": Task003_Liver_sys,
} # map `Task003_Liver` to label_sys `Task003_Liver_sys`
'''


Task003_Liver_sys = { 
#    "0": "background", 
   "agg:1,2": "liver", 
#    "1": "liver", 
#    "2": "cancer"
}
Task007_Pancreas_sys = {
#    "0": "background",
   "agg:1,2": "pancreas",
#    "1": "pancreas",
#    "2": "cancer"
 }
Task009_Spleen_sys = {
#    "0": "background",
   "1": "spleen"
}
FLARE22_sys = {
    # "0":  "background",
    "1":  "liver",
    "10": "esophagus",
    "11": "stomach",
    "12": "duodenum",
    "13": "left kidney",
    "2":  "right kidney",
    "3":  "spleen",
    "4":  "pancreas",
    "5":  "aorta",
    "6":  "IVC",
    "7":  "RAG",
    "8":  "LAG",
    "9":  "gallbladder"
}
AMOS_sys = {
    # "0": "background", 
    "1": "spleen", 
    "2": "right kidney", 
    "3": "left kidney", 
    # "4": "gallbladder", 
    "5": "esophagus", 
    "6": "liver", 
    "7": "stomach", 
    "8": "aorta", 
    # "9": "postcava",
    "10": "pancreas", 
    "11": "RAG", 
    "12": "LAG", 
    "13": "duodenum", 
    "14": "bladder", 
    # "15": "prostate/uterus"
}
Task011_BTCV_sys = {
    # "00": "background",
    "1": "spleen",
    "2": "right kidney",
    "3": "left kidney",
    "4": "gallbladder",
    "5": "esophagus",
    "6": "liver",
    "7": "stomach",
    "8": "aorta",
    "9": "inferior vena cava",
    "10": "portal vein and splenic vein",
    "11": "pancreas",
    "12": "right adrenal gland",
    "13": "left adrenal gland"
}
Task020_AbdomenCT1K_sys = {
    # "0": "background",
    "1": "liver",
    "2": "kidney",
    "3": "spleen",
    "4": "pancreas"
}
Task021_KiTS2021_sys = {
    # "0": "background",
    "agg:1,2,3": "kidney", 
    # "1": "kidney",
    # "2": "tumor",
    # "3": "cyst"
}
Task083_VerSe2020_sys = {
    # "0": "0",
    "1":    "C1",
    "2":    "C2",
    "3":    "C3",
    "4":    "C4",
    "5":    "C5",
    "6":    "C6",
    "7":    "C7",
    "8":    "T1",
    "9":    "T2",
    "10":   "T3",
    "11":   "T4",
    "12":   "T5",
    "13":   "T6",
    "14":   "T7",
    "15":   "T8",
    "16":   "T9",
    "17":   "T10",
    "18":   "T11",
    "19":   "T12",
    "20":   "L1",
    "21":   "L2",
    "22":   "L3",
    "23":   "L4",
    "24":   "L5",
    # "25": "L6",
    # "26": "sacrum",
    # "27": "cocygis",
    # "28": "T13"
}
Task030_CT_ORG_sys = {
    # "0": "background",
    "1": "liver",
    "2": "bladder",
    "3": "lung",
    "4": "kidney",
    # "5": "bone",
    "6": "brain"
}
Task012_BTCV_Cervix_sys = {
    # "0": "background",
    "1": "urinary_bladder",
    # "2": "uterus",
    # "3": "rectum",
    # "4": " small bowel"
}
Task040_KiTS_sys = {
    # "0": 0,
    "1": "kidney",
    # "2": "renal_tumor"
}
Task036_KiPA22_sys = {
    # "0": 0,
    # "1": "kidney_veins",
    "2": "kidney",
    # "3": "kidney_arteries",
    # "4": "renal_tumor"
}
Task560_WORD_sys = {
    # "0": "background",
    "1": "liver",
    "10": "colon",
    # "11": "intestine",
    # "12": "adrenal",
    # "13": "rectum",
    "14": "urinary_bladder",
    "15": "femur_left",
    "16": "femur_right",
    "2": "spleen",
    "3": "kidney_left",
    "4": "kidney_right",
    "5": "stomach",
    "6": "gallbladder",
    "7": "esophagus",
    "8": "pancreas",
    "9": "duodenum"
}
Task605_SegThor_sys = {
    # "0": "background",
    "1": "esophagus",
    # "2": "heart",
    "3": "trachea",
    "4": "aorta"
}
 
label_mapping = {
    "lung":                         "agg:lung_lower_lobe_left,lung_lower_lobe_right,lung_middle_lobe_right,lung_upper_lobe_left,lung_upper_lobe_right",
    "liver":                        "liver",
    "esophagus":                    "esophagus",
    "stomach":                      "stomach",
    "duodenum":                     "duodenum", 
    "kidney":                       "agg:kidney_left,kidney_right",
    "left kidney":                  "kidney_left",
    "right kidney":                 "kidney_right",
    "spleen":                       "spleen",
    "pancreas":                     "pancreas",
    "aorta":                        "aorta",
    "IVC":                          "inferior_vena_cava",
    "RAG":                          "adrenal_gland_right",
    "LAG":                          "adrenal_gland_left",
    "gallbladder":                  "gallbladder",
    "bladder":                      "urinary_bladder",
    "LV":                           "heart_ventricle_left",
    "RV":                           "heart_ventricle_right",
    "LA":                           "heart_atrium_left",
    "brain":                        "brain",
    "inferior vena cava":           "inferior_vena_cava",
    "portal vein and splenic vein": "portal_vein_and_splenic_vein",
    "right adrenal gland":          "adrenal_gland_right",
    "left adrenal gland":           "adrenal_gland_left",
    "C1":                           "vertebrae_C1",
    "C2":                           "vertebrae_C2",
    "C3":                           "vertebrae_C3",
    "C4":                           "vertebrae_C4",
    "C5":                           "vertebrae_C5",
    "C6":                           "vertebrae_C6",
    "C7":                           "vertebrae_C7",
    "T1":                           "vertebrae_T1",
    "T2":                           "vertebrae_T2",
    "T3":                           "vertebrae_T3",
    "T4":                           "vertebrae_T4",
    "T5":                           "vertebrae_T5",
    "T6":                           "vertebrae_T6",
    "T7":                           "vertebrae_T7",
    "T8":                           "vertebrae_T8",
    "T9":                           "vertebrae_T9",
    "T10":                          "vertebrae_T10",
    "T11":                          "vertebrae_T11",
    "T12":                          "vertebrae_T12",
    "L1":                           "vertebrae_L1",
    "L2":                           "vertebrae_L2",
    "L3":                           "vertebrae_L3",
    "L4":                           "vertebrae_L4",
    "L5":                           "vertebrae_L5",
}

label_sys_dict = {
    "Task003_Liver":                        Task003_Liver_sys,
    "Task007_Pancreas":                     Task007_Pancreas_sys,
    "Task009_Spleen":                       Task009_Spleen_sys,
    "Task011_BTCV":                         Task011_BTCV_sys,
    "Task012_BTCV_Cervix":                  Task012_BTCV_Cervix_sys,
    "Task020_AbdomenCT1K":                  Task020_AbdomenCT1K_sys,
    "Task021_KiTS2021":                     Task021_KiTS2021_sys,  
    "Task022_FLARE22":                      FLARE22_sys,
    "Task030_CT_ORG":                       Task030_CT_ORG_sys,
    "Task032_AMOS22_Task1":                 AMOS_sys,
    "Task036_KiPA22":                       Task036_KiPA22_sys,  
    "Task083_VerSe2020":                    Task083_VerSe2020_sys,   
    "Task560_WORD":                         Task560_WORD_sys,             
    "Task605_SegThor":                      Task605_SegThor_sys,        
}

totalseg_json = {
    "1": "adrenal_gland_left",
    "10": "duodenum",
    "100": "vertebrae_T5",
    "101": "vertebrae_T6",
    "102": "vertebrae_T7",
    "103": "vertebrae_T8",
    "104": "vertebrae_T9",
    "11": "esophagus",
    "12": "face",
    "13": "femur_left",
    "14": "femur_right",
    "15": "gallbladder",
    "16": "gluteus_maximus_left",
    "17": "gluteus_maximus_right",
    "18": "gluteus_medius_left",
    "19": "gluteus_medius_right",
    "2": "adrenal_gland_right",
    "20": "gluteus_minimus_left",
    "21": "gluteus_minimus_right",
    "22": "heart_atrium_left",
    "23": "heart_atrium_right",
    "24": "heart_myocardium",
    "25": "heart_ventricle_left",
    "26": "heart_ventricle_right",
    "27": "hip_left",
    "28": "hip_right",
    "29": "humerus_left",
    "3": "aorta",
    "30": "humerus_right",
    "31": "iliac_artery_left",
    "32": "iliac_artery_right",
    "33": "iliac_vena_left",
    "34": "iliac_vena_right",
    "35": "iliopsoas_left",
    "36": "iliopsoas_right",
    "37": "inferior_vena_cava",
    "38": "kidney_left",
    "39": "kidney_right",
    "4": "autochthon_left",
    "40": "liver",
    "41": "lung_lower_lobe_left",
    "42": "lung_lower_lobe_right",
    "43": "lung_middle_lobe_right",
    "44": "lung_upper_lobe_left",
    "45": "lung_upper_lobe_right",
    "46": "pancreas",
    "47": "portal_vein_and_splenic_vein",
    "48": "pulmonary_artery",
    "49": "rib_left_1",
    "5": "autochthon_right",
    "50": "rib_left_10",
    "51": "rib_left_11",
    "52": "rib_left_12",
    "53": "rib_left_2",
    "54": "rib_left_3",
    "55": "rib_left_4",
    "56": "rib_left_5",
    "57": "rib_left_6",
    "58": "rib_left_7",
    "59": "rib_left_8",
    "6": "brain",
    "60": "rib_left_9",
    "61": "rib_right_1",
    "62": "rib_right_10",
    "63": "rib_right_11",
    "64": "rib_right_12",
    "65": "rib_right_2",
    "66": "rib_right_3",
    "67": "rib_right_4",
    "68": "rib_right_5",
    "69": "rib_right_6",
    "7": "clavicula_left",
    "70": "rib_right_7",
    "71": "rib_right_8",
    "72": "rib_right_9",
    "73": "sacrum",
    "74": "scapula_left",
    "75": "scapula_right",
    "76": "small_bowel",
    "77": "spleen",
    "78": "stomach",
    "79": "trachea",
    "8": "clavicula_right",
    "80": "urinary_bladder",
    "81": "vertebrae_C1",
    "82": "vertebrae_C2",
    "83": "vertebrae_C3",
    "84": "vertebrae_C4",
    "85": "vertebrae_C5",
    "86": "vertebrae_C6",
    "87": "vertebrae_C7",
    "88": "vertebrae_L1",
    "89": "vertebrae_L2",
    "9": "colon",
    "90": "vertebrae_L3",
    "91": "vertebrae_L4",
    "92": "vertebrae_L5",
    "93": "vertebrae_T1",
    "94": "vertebrae_T10",
    "95": "vertebrae_T11",
    "96": "vertebrae_T12",
    "97": "vertebrae_T2",
    "98": "vertebrae_T3",
    "99": "vertebrae_T4"
}

totalseg_cls2idx = dict()
for k,v in totalseg_json.items():
    totalseg_cls2idx[v] = int(k)
# print(totalseg_cls2idx)