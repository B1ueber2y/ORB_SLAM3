import sys, os
from scipy.spatial.transform import Rotation as Rot
import numpy as np
from collections import defaultdict
from kornia.geometry.conversions import camtoworld_graphics_to_vision_Rt
from torch import Tensor

def read_pose(file, poses_dict):
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            name = int(float(line.split()[0]))
            pose = line.split()[1:]
            if name not in poses_dict:
                poses_dict[name] = pose

kf_pose_file = sys.argv[1]
f_pose_file = sys.argv[2]
out_dir = sys.argv[3]

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

poses = {}
read_pose(kf_pose_file, poses)
read_pose(f_pose_file, poses)

out_lines = []
num_obs = 0
with open('images_tmp.txt') as f:
    lines = f.readlines()
    cnt = 0
    for line in lines:
        name = line.split()[8].split('/')[-1].split('.')[0]
        if int(name.split('_')[0]) not in poses:
            continue

        # Convert the pose
        correct_pose = poses[int(name.split('_')[0])]
        R = Rot.from_quat([float(s) for s in correct_pose[3:7]])
        t = np.array([float(s) for s in correct_pose[0:3]])

        # From SLAM coordinate to COLMAP
        # visionR, visionT = camtoworld_graphics_to_vision_Rt(Tensor(R.as_matrix()[None, :, :]), Tensor(t[None, :, None]))
        # nR = Rot.from_matrix(visionR[0])

        # From cam2world to world2cam
        nq = R.inv().as_quat()
        nt = -R.inv().as_matrix() @ t
        
        p = [nq[3], nq[0], nq[1], nq[2], nt[0], nt[1], nt[2]]
        p = [str(s) for s in p]

        out_l = str(cnt) + ' ' + ' '.join(p + [line.split()[7], line.split()[8].split('/')[-1]]) + '\n'
        out_l += ' '.join(line.split()[9:]) + '\n'
        num_obs += len(line.split()[9:]) // 3
        out_lines.append(out_l)
        cnt += 1
    
mean_obs = num_obs / cnt
HEADER = "# Image list with two lines of data per image:\n" + \
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n" + \
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n" + \
        "# Number of images: {}, mean observations per image: {}\n".format(cnt, mean_obs)
with open(os.path.join(out_dir, 'images.txt'), 'w') as f:
    f.write(HEADER)
    f.writelines(out_lines)

point_tracks = defaultdict(list)
with open(os.path.join(out_dir, 'images.txt')) as f:
    lines = f.readlines()[4:]
    for i in range(0, len(lines), 2):
        img_id = int(lines[i].split()[0])
        elems = lines[i+1].split()
        point3D_ids = np.array(tuple(map(int, elems[2::3])))
        for point2d_idx in range(len(point3D_ids)):
            p3d_id = point3D_ids[point2d_idx]
            point_tracks[p3d_id].append((img_id, point2d_idx))

out_lines = []
num_tracks = 0

R = Tensor(np.eye(3)[None, :, :])
t = Tensor(np.array([0, 0, 0])[None, :, None])
R, t = camtoworld_graphics_to_vision_Rt(R, t)
R = R[0].numpy()

with open('points3D_tmp.txt') as f:
    lines = f.readlines()
    r, g, b, err = 0, 0, 0, 0
    for line in lines:
        p3d_id = int(line.split()[0])
        xyz = np.array([float(s) for s in line.split()[1:4]])
        # xyz = xyz @ R
        l = [str(s) for s in [p3d_id] + list(xyz)]
        for img_id, p2d_idx in point_tracks[p3d_id]:
            l.append(str(img_id))
            l.append(str(p2d_idx))
        if len(point_tracks[p3d_id]) > 0:
            num_tracks += len(point_tracks[p3d_id])
            out_lines.append(' '.join(l) + '\n')

mean_track_length = num_tracks / len(out_lines)
HEADER = "# 3D point list with one line of data per point:\n" + \
         "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n" + \
         "# Number of points: {}, mean track length: {}\n".format(len(out_lines), mean_track_length)
with open(os.path.join(out_dir, 'points3D.txt'), 'w') as f:
    f.write(HEADER)
    f.writelines(out_lines)