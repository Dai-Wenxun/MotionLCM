# Map joints Name to SMPL joints idx
JOINT_MAP = {
    'MidHip': 0,
    'LHip': 1, 'LKnee': 4, 'LAnkle': 7, 'LFoot': 10,
    'RHip': 2, 'RKnee': 5, 'RAnkle': 8, 'RFoot': 11,
    'LShoulder': 16, 'LElbow': 18, 'LWrist': 20, 'LHand': 22,
    'RShoulder': 17, 'RElbow': 19, 'RWrist': 21, 'RHand': 23,
    'spine1': 3, 'spine2': 6, 'spine3': 9, 'Neck': 12, 'Head': 15,
    'LCollar': 13, 'RCollar': 14
}

full_smpl_idx = range(24)
key_smpl_idx = [0, 1, 4, 7, 2, 5, 8, 17, 19, 21, 16, 18, 20]

AMASS_JOINT_MAP = {
    'MidHip': 0,
    'LHip': 1, 'LKnee': 4, 'LAnkle': 7, 'LFoot': 10,
    'RHip': 2, 'RKnee': 5, 'RAnkle': 8, 'RFoot': 11,
    'LShoulder': 16, 'LElbow': 18, 'LWrist': 20,
    'RShoulder': 17, 'RElbow': 19, 'RWrist': 21,
    'spine1': 3, 'spine2': 6, 'spine3': 9, 'Neck': 12, 'Head': 15,
    'LCollar': 13, 'RCollar': 14,
}
amass_idx = range(22)
amass_smpl_idx = range(22)

SMPL_MODEL_DIR = "./deps/smpl_models"
GMM_MODEL_DIR = "./deps/smpl_models"
SMPL_MEAN_FILE = "./deps/smpl_models/neutral_smpl_mean_params.h5"
# for collision
Part_Seg_DIR = "./deps/smpl_models/smplx_parts_segm.pkl"
