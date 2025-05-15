from joblib import Parallel, delayed
import numpy as np
from pathlib import Path
import os

def rotate_points_and_normals_np(xyz, normals, R, center=None, normalize_normals=True):
    assert xyz.shape[1] == 3
    assert normals.shape[1] == 3
    assert R.shape == (3, 3)
    if center is not None:
        xyz_centered = xyz - center
    else:
        xyz_centered = xyz
    xyz_rot = xyz_centered @ R.T
    if center is not None:
        xyz_rot += center

    normals_rot = np.empty_like(normals)
    normal_lengths = np.linalg.norm(normals, axis=-1)
    valid_normals = normal_lengths > 1e-6
    normals_rot[valid_normals] = normals[valid_normals] @ R.T
    normals_rot[~valid_normals] = normals[~valid_normals]

    if normalize_normals:
        norms = np.linalg.norm(normals_rot, axis=-1, keepdims=True) + 1e-8
        normals_rot /= norms
    return xyz_rot, normals_rot

def np_rotation_matrix_z(angle_deg):
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return np.array([
        [cos_t, -sin_t, 0.0],
        [sin_t,  cos_t, 0.0],
        [0.0,    0.0,   1.0]
    ])

def process_sample(root, s):
    try:
        orient = int(s.split('_')[2])
        parent = os.path.join(root, '_'.join(s.split('_')[0:2]))
        child = os.path.join(parent, s)

        path_x = os.path.join(parent, 'x.npy')
        path_y = os.path.join(child, 'y.npy')
        path_xrot = os.path.join(child, 'x_rot.npy')
        path_posrot = os.path.join(child, 'pos_rot.npy')
        path_stats_x = os.path.join(child, 'stats_x.npy')
        path_stats_y = os.path.join(child, 'stats_y.npy')
        path_stats_ct = os.path.join(child, 'stats_ct.npy')

        if not (os.path.exists(path_x) and os.path.exists(path_y)):
            return

        if all(os.path.exists(p) for p in [path_xrot, path_posrot, path_stats_x, path_stats_y, path_stats_ct]):
            return  

        x = np.load(path_x)
        y = np.load(path_y)

        xyz = x[:, :3]
        normals = x[:, 4:]
        R = np_rotation_matrix_z(orient)
        center = np.array([0, 0, 0])
        xyz_rot, normals_rot = rotate_points_and_normals_np(xyz, normals, R, center=center)
        x[:, :3] = xyz_rot
        x[:, 4:] = normals_rot

        np.save(path_xrot, x)
        np.save(path_posrot, xyz)

        xy = np.concatenate([x, y], axis=1)
        valid_mask = ~np.isnan(xy).any(axis=1)
        x = x[valid_mask]
        y = y[valid_mask]

        np.save(path_stats_x, x.sum(axis=0))
        np.save(path_stats_y, y.sum(axis=0))
        np.save(path_stats_ct, x.shape[0])
        print(f"Processed {s}")

    except Exception as e:
        print(f"Error processing {s}: {e}")


vtknpy_dir = # Update your folder structure and directories
root = Path(vtknpy_dir)
samples = [p.relative_to(root).name for p in root.rglob('*') if p.is_dir() and len(p.relative_to(root).parts) == 2]

Parallel(n_jobs=8, backend='multiprocessing')(
    delayed(process_sample)(str(root), s) for s in samples
)
