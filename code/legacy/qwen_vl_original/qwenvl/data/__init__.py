import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

TRY = {
    "annotation_path": "/wangbenyou-dengyizhe/embody-3d/data/conversations_train.json",
    "data_path": "/wangbenyou-dengyizhe/embody-3d/data",
}

TRY_TEST = {
    "annotation_path": "/wangbenyou-dengyizhe/embody-3d/data/conversations_test.json",
    "data_path": "/wangbenyou-dengyizhe/embody-3d/data",
}

VIDEO_COT = {
    "annotation_path": "/wangbenyou-dengyizhe/data/video-cot/videoespresso_rabbids_2k_processed.json",
    "data_path": "/wangbenyou-dengyizhe/data/video-cot"
}

MOTIONX_V0 = {
    "annotation_path": "/wangbenyou-dengyizhe/Data/processed/motionx_overall_action_conversations_v0.json",
    "data_path": "/wangbenyou-dengyizhe/Data"
}

MOTIONX_V0_1 = {
    "annotation_path": "/wangbenyou-dengyizhe/Data/processed/motionx_overall_action_conversations_v0_1.json",
    "data_path": "/wangbenyou-dengyizhe/Data"
}

# 训练集：划分后的训练部分（需先运行 split_motionx_data.py）
MOTIONX_V0_1_TRAIN = {
    "annotation_path": "/wangbenyou-dengyizhe/Data/processed/motionx_overall_action_conversations_v0_1_train.json",
    "data_path": "/wangbenyou-dengyizhe/Data"
}

# 测试集：划分后的 100 条
MOTIONX_V0_1_TEST = {
    "annotation_path": "/wangbenyou-dengyizhe/Data/processed/motionx_overall_action_conversations_v0_1_test.json",
    "data_path": "/wangbenyou-dengyizhe/Data"
}

MOTIONX_V1_0_TRAIN = {
    "annotation_path": "/wangbenyou-dengyizhe/Data/processed/motionx_overall_action_conversations_v1_0_train.json",
    "data_path": "/wangbenyou-dengyizhe/Data"
}

MOTIONX_V1_0_TEST = {
    "annotation_path": "/wangbenyou-dengyizhe/Data/processed/motionx_overall_action_conversations_v1_0_test.json",
    "data_path": "/wangbenyou-dengyizhe/Data"
}

MOTIONX_V1_0_MOTION_ONLY = {
    "annotation_path": "/wangbenyou-dengyizhe/Data/processed/motionx_overall_action_conversations_v1_0_train_motion_only.json",
    "data_path": "/wangbenyou-dengyizhe/Data"
}

MOTIONX_V1_0_VIDEO_ONLY = {
    "annotation_path": "/wangbenyou-dengyizhe/Data/processed/motionx_overall_action_conversations_v1_0_train_video_only.json",
    "data_path": "/wangbenyou-dengyizhe/Data"
}

MOTIONX_V1_1_TRAIN = {
    "annotation_path": "/wangbenyou-dengyizhe/Data/processed/motionx_overall_action_conversations_v1_1_train.json",
    "data_path": "/wangbenyou-dengyizhe/Data"
}

MOTIONX_V1_1_MOTION_ONLY = {
    "annotation_path": "/wangbenyou-dengyizhe/Data/processed/motionx_overall_action_conversations_v1_1_train_motion_only.json",
    "data_path": "/wangbenyou-dengyizhe/Data"
}

MOTIONX_V1_1_TEST = {
    "annotation_path": "/wangbenyou-dengyizhe/Data/processed/motionx_overall_action_conversations_v1_1_test.json",
    "data_path": "/wangbenyou-dengyizhe/Data"
}

CHARADES_V1_1_TRAIN = {
    "annotation_path": "/wangbenyou-dengyizhe/Data/processed/charades_overall_action_conversations_v1_1_train.json",
    "data_path": "/wangbenyou-dengyizhe/Data"
}

CHARADES_V1_1_MOTION_ONLY = {
    "annotation_path": "/wangbenyou-dengyizhe/Data/processed/charades_overall_action_conversations_v1_1_train_motion_only.json",
    "data_path": "/wangbenyou-dengyizhe/Data"
}

HUMANML3D_V1_1_TRAIN_MOTION_ONLY = {
    "annotation_path": "/wangbenyou-dengyizhe/Data/processed/HumanML3D_overall_action_conversations_v1_1_train_motion_only.json",
    "data_path": "/wangbenyou-dengyizhe/Data"
}

SHAREGPT_V1_1_TRAIN_TEXT_ONLY= {
    "annotation_path": "/wangbenyou-dengyizhe/Data/processed/sharegpt_single_turn_conversations_v1_1_train_text_only.json",
    "data_path": "/wangbenyou-dengyizhe/Data"
}

LMSYS_CHAT_1M_V1_1_TRAIN_TEXT_ONLY= {
    "annotation_path": "/wangbenyou-dengyizhe/Data/processed/lmsys_single_turn_conversations_v1_1_train_text_only.json",
    "data_path": "/wangbenyou-dengyizhe/Data"
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "try": TRY,
    "try_test": TRY_TEST,
    "video_cot": VIDEO_COT,
    "motionX_v0": MOTIONX_V0,
    "motionX_v0_1": MOTIONX_V0_1,
    "motionX_v0_1_train": MOTIONX_V0_1_TRAIN,
    "motionX_v0_1_test": MOTIONX_V0_1_TEST,
    "motionX_v1_0_train": MOTIONX_V1_0_TRAIN,
    "motionX_v1_0_test": MOTIONX_V1_0_TEST,
    "motionX_v1_0_motion_only": MOTIONX_V1_0_MOTION_ONLY,
    "motionX_v1_0_video_only": MOTIONX_V1_0_VIDEO_ONLY,
    "motionX_v1_1_train": MOTIONX_V1_1_TRAIN,
    "motionX_v1_1_test": MOTIONX_V1_1_TEST,
    "motionX_v1_1_motion_only": MOTIONX_V1_1_MOTION_ONLY,
    "charades_v1_1_train": CHARADES_V1_1_TRAIN,
    "charades_v1_1_motion_only": CHARADES_V1_1_MOTION_ONLY,
    "humanml3d_v1_1_train_motion_only": HUMANML3D_V1_1_TRAIN_MOTION_ONLY,
    "sharegpt_v1_1_train_text_only": SHAREGPT_V1_1_TRAIN_TEXT_ONLY,
    "lmsys_chat_1m_v1_1_train_text_only": LMSYS_CHAT_1M_V1_1_TRAIN_TEXT_ONLY,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
