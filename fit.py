# borrow from optimization https://github.com/wangsen1312/joints2smpl

import argparse
import pickle

import h5py
import smplx

import torch

from mld.transforms.joints2rots import config
from mld.transforms.joints2rots.smplify import SMPLify3D

parser = argparse.ArgumentParser()
parser.add_argument("--pkl", type=str, required=True, help="path of data pkl file")
parser.add_argument("--num_smplify_iters", type=int, default=150, help="num of smplify iters")
parser.add_argument("--cuda", type=bool, default=True, help="enables cuda")
parser.add_argument("--gpu_ids", type=int, default=0, help="choose gpu ids")
parser.add_argument("--num_joints", type=int, default=22, help="joint number")
parser.add_argument("--joint_category", type=str, default="AMASS", help="use correspondence")
parser.add_argument("--fix_foot", type=str, default="False", help="fix foot or not")
opt = parser.parse_args()
print(opt)

# load joints
with open(opt.pkl, 'rb') as f:
    data = pickle.load(f)
    if 'vertices' in data:
        print('mesh already rendered!')
        exit(-1)

joints = data['joints']

# load predefined something
device = torch.device("cuda:" + str(opt.gpu_ids) if opt.cuda else "cpu")
print(config.SMPL_MODEL_DIR)
smplxmodel = smplx.create(
    config.SMPL_MODEL_DIR,
    model_type="smpl",
    gender="neutral",
    ext="pkl",
    batch_size=joints.shape[0],
).to(device)

# load the mean pose as original
smpl_mean_file = config.SMPL_MEAN_FILE

file = h5py.File(smpl_mean_file, "r")
init_mean_pose = (
    torch.from_numpy(file["pose"][:])
    .unsqueeze(0).repeat(joints.shape[0], 1)
    .float()
    .to(device)
)
init_mean_shape = (
    torch.from_numpy(file["shape"][:])
    .unsqueeze(0).repeat(joints.shape[0], 1)
    .float()
    .to(device)
)
cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(device)

# initialize SMPLify
smplify = SMPLify3D(
    smplxmodel=smplxmodel,
    batch_size=joints.shape[0],
    joints_category=opt.joint_category,
    num_iters=opt.num_smplify_iters,
    device=device,
)
print("initialize SMPLify3D done!")

print("Start SMPLify!")
keypoints_3d = torch.Tensor(joints).to(device).float()

if opt.joint_category == "AMASS":
    confidence_input = torch.ones(opt.num_joints)
    # make sure the foot and ankle
    if opt.fix_foot:
        confidence_input[7] = 1.5
        confidence_input[8] = 1.5
        confidence_input[10] = 1.5
        confidence_input[11] = 1.5
else:
    print("Such category not settle down!")

# ----- from initial to fitting -------
(
    new_opt_vertices,
    new_opt_joints,
    new_opt_pose,
    new_opt_betas,
    new_opt_cam_t,
    new_opt_joint_loss,
) = smplify(
    init_mean_pose.detach(),
    init_mean_shape.detach(),
    cam_trans_zero.detach(),
    keypoints_3d,
    conf_3d=confidence_input.to(device)
)

# fix shape
betas = torch.zeros_like(new_opt_betas)
root = keypoints_3d[:, 0, :]

output = smplxmodel(
    betas=betas,
    global_orient=new_opt_pose[:, :3],
    body_pose=new_opt_pose[:, 3:],
    transl=root,
    return_verts=True,
)
vertices = output.vertices.detach().cpu().numpy()
data['vertices'] = vertices

with open(opt.pkl, 'wb') as f:
    pickle.dump(data, f)
print(f'vertices saved in {opt.pkl}')
