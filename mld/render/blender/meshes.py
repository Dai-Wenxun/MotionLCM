import numpy as np

from .materials import body_material

# Orange
GEN_SMPL = body_material(0.658, 0.214, 0.0114)
# Green
GT_SMPL = body_material(0.035, 0.415, 0.122)


class Meshes:
    def __init__(self, data, gt, mode, trajectory, faces_path, always_on_floor, oldrender=True):
        data, trajectory = prepare_meshes(data, trajectory, always_on_floor=always_on_floor)

        self.faces = np.load(faces_path)
        print(faces_path)
        self.data = data
        self.mode = mode
        self.oldrender = oldrender

        self.N = len(data)
        # self.trajectory = data[:, :, [0, 1]].mean(1)
        self.trajectory = trajectory

        if gt:
            self.mat = GT_SMPL
        else:
            self.mat = GEN_SMPL

    def get_sequence_mat(self, frac):
        import matplotlib
        # cmap = matplotlib.cm.get_cmap('Blues')
        cmap = matplotlib.cm.get_cmap('Oranges')
        # begin = 0.60
        # end = 0.90
        begin = 0.50
        end = 0.90
        rgb_color = cmap(begin + (end-begin)*frac)
        mat = body_material(*rgb_color, oldrender=self.oldrender)
        return mat

    def get_root(self, index):
        return self.data[index].mean(0)

    def get_mean_root(self):
        return self.data.mean((0, 1))

    def load_in_blender(self, index, mat):
        vertices = self.data[index]
        faces = self.faces
        name = f"{str(index).zfill(4)}"

        from .tools import load_numpy_vertices_into_blender
        load_numpy_vertices_into_blender(vertices, faces, name, mat)

        return name

    def __len__(self):
        return self.N


def prepare_meshes(data, trajectory, always_on_floor=False):
    # Swap axis (gravity=Z instead of Y)
    data = data[..., [2, 0, 1]]

    if trajectory is not None:
        trajectory = trajectory[..., [2, 0, 1]]
        mask = trajectory.sum(-1) != 0
        trajectory = trajectory[mask]

    # Remove the floor
    height_offset = data[..., 2].min()
    data[..., 2] -= height_offset

    if trajectory is not None:
        trajectory[..., 2] -= height_offset

    # Put all the body on the floor
    if always_on_floor:
        data[..., 2] -= data[..., 2].min(1)[:, None]

    return data, trajectory
