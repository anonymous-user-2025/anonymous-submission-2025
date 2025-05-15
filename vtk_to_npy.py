# Adapted from: https://github.com/thuml/Neural-Solver-Library, Neural-Solver-Library/data_provider/shapenet_utils.py
# Modified: adapted to data structure and outputs, SDF calculated directly from geometry
# Licensed under the MIT License
# © Original Author(s): Haixu Wu and Huakun Luo and Haowen Wang and Jianmin Wang and Mingsheng Long
# Associated paper: Wu, H., Luo, H., Wang, H., Wang, J. and Long, M., 2024. Transolver: A fast transformer solver for pdes on general geometries. arXiv preprint arXiv:2402.02366.

import torch
import vtk
import os
import itertools
import random
import numpy as np
import igl
import trimesh
import torch_geometric
from torch_geometric import nn as nng
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import k_hop_subgraph, subgraph
from vtk.util.numpy_support import vtk_to_numpy
from pathlib import Path
import gc
import psutil


def load_unstructured_grid_data(file_name):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()
    output = reader.GetOutput()
    return output

def load_poly_data(file_name):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    output = reader.GetOutput()
    return output


def unstructured_grid_data_to_poly_data(unstructured_grid_data):
    filter = vtk.vtkDataSetSurfaceFilter()
    filter.SetInputData(unstructured_grid_data)
    filter.Update()
    poly_data = filter.GetOutput()
    return poly_data, filter


def get_sdf(target, boundary):
    nbrs = NearestNeighbors(n_neighbors=1).fit(boundary)
    dists, indices = nbrs.kneighbors(target)
    neis = np.array([boundary[i[0]] for i in indices])
    dirs = (target - neis) / (dists + 1e-8)
    return dists.reshape(-1), dirs


def get_normal(unstructured_grid_data):
    poly_data, surface_filter = unstructured_grid_data_to_poly_data(unstructured_grid_data)
    # visualize_poly_data(poly_data, surface_filter)
    # poly_data.GetPointData().SetScalars(None)
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(poly_data)
    normal_filter.SetAutoOrientNormals(1)
    normal_filter.SetConsistency(1)
    # normal_filter.SetSplitting(0)
    normal_filter.SetComputeCellNormals(1)
    normal_filter.SetComputePointNormals(0)
    normal_filter.Update()
    '''
    normal_filter.SetComputeCellNormals(0)
    normal_filter.SetComputePointNormals(1)
    normal_filter.Update()
    #visualize_poly_data(poly_data, surface_filter, normal_filter)
    poly_data.GetPointData().SetNormals(normal_filter.GetOutput().GetPointData().GetNormals())
    p2c = vtk.vtkPointDataToCellData()
    p2c.ProcessAllArraysOn()
    p2c.SetInputData(poly_data)
    p2c.Update()
    unstructured_grid_data.GetCellData().SetNormals(p2c.GetOutput().GetCellData().GetNormals())
    #visualize_poly_data(poly_data, surface_filter, p2c)
    '''

    unstructured_grid_data.GetCellData().SetNormals(normal_filter.GetOutput().GetCellData().GetNormals())
    c2p = vtk.vtkCellDataToPointData()
    # c2p.ProcessAllArraysOn()
    c2p.SetInputData(unstructured_grid_data)
    c2p.Update()
    unstructured_grid_data = c2p.GetOutput()
    # return unstructured_grid_data
    normal = vtk_to_numpy(c2p.GetOutput().GetPointData().GetNormals()).astype(np.double)
    # print(np.max(np.max(np.abs(normal), axis=1)), np.min(np.max(np.abs(normal), axis=1)))
    normal /= (np.max(np.abs(normal), axis=1, keepdims=True) + 1e-8)
    normal /= (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8)
    if np.isnan(normal).sum() > 0:
        print(np.isnan(normal).sum())
        print("recalculate")
        return get_normal(unstructured_grid_data)  # re-calculate
    # print(normal)
    return normal


def get_edges(unstructured_grid_data, points, cell_size=4):
    edge_indeces = set()
    cells = vtk_to_numpy(unstructured_grid_data.GetCells().GetData()).reshape(-1, cell_size + 1)
    for i in range(len(cells)):
        for j, k in itertools.product(range(1, cell_size + 1), repeat=2):
            edge_indeces.add((cells[i][j], cells[i][k]))
            edge_indeces.add((cells[i][k], cells[i][j]))
    edges = [[], []]
    for u, v in edge_indeces:
        edges[0].append(tuple(points[u]))
        edges[1].append(tuple(points[v]))
    return edges

def get_edges(unstructured_grid_data, points):
    edge_indices = set()
    cell_array = unstructured_grid_data.GetPolys() if isinstance(unstructured_grid_data, vtk.vtkPolyData) \
                 else unstructured_grid_data.GetCells()

    id_list = vtk.vtkIdList()
    for i in range(unstructured_grid_data.GetNumberOfCells()):
        unstructured_grid_data.GetCellPoints(i, id_list)
        ids = [id_list.GetId(j) for j in range(id_list.GetNumberOfIds())]
        for u, v in itertools.permutations(ids, 2):
            edge_indices.add((u, v))

    edges = [[], []]
    for u, v in edge_indices:
        edges[0].append(tuple(points[u]))
        edges[1].append(tuple(points[v]))
    return edges



def get_edge_index(pos, edges_press, edges_velo):
    indices = {tuple(pos[i]): i for i in range(len(pos))}
    edges = set()
    for i in range(len(edges_press[0])):
        edges.add((indices[edges_press[0][i]], indices[edges_press[1][i]]))
    for i in range(len(edges_velo[0])):
        edges.add((indices[edges_velo[0][i]], indices[edges_velo[1][i]]))
    edge_index = np.array(list(edges)).T
    return edge_index


def get_induced_graph(data, idx, num_hops):
    subset, sub_edge_index, _, _ = k_hop_subgraph(node_idx=idx, num_hops=num_hops, edge_index=data.edge_index,
                                                  relabel_nodes=True)
    return Data(x=data.x[subset], y=data.y[idx], edge_index=sub_edge_index)


def pc_normalize(pc):
    centroid = torch.mean(pc, axis=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def get_shape(data, max_n_point=8192, normalize=True, use_height=False):
    surf_indices = torch.where(data.surf)[0].tolist()

    if len(surf_indices) > max_n_point:
        surf_indices = np.array(random.sample(range(len(surf_indices)), max_n_point))

    shape_pc = data.pos[surf_indices].clone()

    if normalize:
        shape_pc = pc_normalize(shape_pc)

    if use_height:
        gravity_dim = 1
        height_array = shape_pc[:, gravity_dim:gravity_dim + 1] - shape_pc[:, gravity_dim:gravity_dim + 1].min()
        shape_pc = torch.cat((shape_pc, height_array), axis=1)

    return shape_pc


def create_edge_index_radius(data, r, max_neighbors=32):
    data.edge_index = nng.radius_graph(x=data.pos, r=r, loop=True, max_num_neighbors=max_neighbors)
    return data

def get_sdf_from_mesh(target_pts, surface_mesh):
    
    sdf, _, _, normals = igl.signed_distance(
        target_pts,
        surface_mesh.vertices,
        surface_mesh.faces.astype(np.int32),
        return_normals=True
    )
    return sdf, normals


class GraphDataset(Dataset):
    def __init__(self, datalist, use_height=False, use_cfd_mesh=True, r=None, coef_norm=None, valid_list=None):
        super().__init__()
        self.datalist = datalist
        self.use_height = use_height
        self.coef_norm = coef_norm
        self.valid_list = valid_list
        if not use_cfd_mesh:
            assert r is not None
            for i in range(len(self.datalist)):
                self.datalist[i] = create_edge_index_radius(self.datalist[i], r)

    def len(self):
        return len(self.datalist)

    def get(self, idx):
        data = self.datalist[idx]
        shape = get_shape(data, use_height=self.use_height)
        if self.valid_list is None:
            return self.datalist[idx].pos, self.datalist[idx].x, self.datalist[idx].y, self.datalist[idx].surf, \
                data.edge_index
        else:
            return self.datalist[idx].pos, self.datalist[idx].x, self.datalist[idx].y, self.datalist[idx].surf, \
                data.edge_index, self.valid_list[idx]


def get_edge_index_from_mesh(unstructured_grid_data, mesh_points, unified_pos):

    mesh_arr = np.ascontiguousarray(mesh_points, dtype=np.float32)
    pos_arr = np.ascontiguousarray(unified_pos, dtype=np.float32)

    def hashable(arr):
        return arr.view([('', arr.dtype)] * arr.shape[1]).ravel()

    mesh_hash = hashable(mesh_arr)
    pos_hash = hashable(pos_arr)

    sort_idx = np.argsort(pos_hash)
    inv_map = np.searchsorted(pos_hash[sort_idx], mesh_hash)
    mapped_indices = sort_idx[inv_map]

    valid = pos_hash[sort_idx][inv_map] == mesh_hash
    mapped_indices[~valid] = -1

    edge_src = []
    edge_dst = []
    id_list = vtk.vtkIdList()

    for i in range(unstructured_grid_data.GetNumberOfCells()):
        unstructured_grid_data.GetCellPoints(i, id_list)
        ids = [id_list.GetId(j) for j in range(id_list.GetNumberOfIds())]
        local = mapped_indices[ids]

        if np.any(local < 0):
            continue

        edge_src.extend(np.repeat(local, len(local)))
        edge_dst.extend(np.tile(local, len(local)))

    edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
    return edge_index


vtk_dir = ## Add your directory for vtk files (and amend folder structure as needed)
vtknpy_dir = ## Add your directory for npy processed files (and amend folder structure as needed)
geo_dir = ## Add your directory for geometry files (and amend folder structure as needed), this is to calculate SDFs from geometry

all_b_names = [f.name for f in Path(vtk_dir).iterdir() if f.is_dir()]

for b in all_b_names:
    
    b_name = b
    available_orient  = None
    
    for orient in (np.arange(8)*45):
        f_name_U = Path(vtk_dir) / b_name / f"{b_name}_{orient}_U.vtk"
        f_name_P = Path(vtk_dir) / b_name / f"{b_name}_{orient}_P.vtk"
        if f_name_U.exists() and f_name_P.exists():
            available_orient = orient
            break 
    if available_orient is None:
        print(f"No valid orientations found for building {b_name}, skipping...")
        continue  
    
    b_dir = available_orient
    f_name_U = vtk_dir + '/' + b_name + '/' + b_name + '_' + str(b_dir) + '_U.vtk'  # Update your folder structure and directories
    f_name_P = vtk_dir + '/' + b_name + '/' + b_name + '_' + str(b_dir) + '_P.vtk'  # Update your folder structure and directories
    save_path_srf = vtknpy_dir + '/' + b_name                                       # Update your folder structure and directories
    
    if(os.path.exists(f_name_U) and os.path.exists(f_name_P) and (not all([os.path.exists(os.path.join(save_path_srf, nn+'.npy')) for nn in ['x','pos','surf']]))):
            
        print('b_name',b_name)
        print('b_dir',b_dir)
        print('f_name_U',f_name_U)
        print('f_name_P',f_name_P)
        print('save_path_srf',save_path_srf)
            
        if (not os.path.exists(vtknpy_dir)):
            os.makedirs(vtknpy_dir)
        if (not os.path.exists(vtknpy_dir + '/' + b_name)):
            os.makedirs(vtknpy_dir + '/' + b_name)
        if (not os.path.exists(vtknpy_dir + '/' + b_name + '/' + b_name + '_' + str(b_dir))):
            os.makedirs(vtknpy_dir + '/' + b_name + '/' + b_name + '_' + str(b_dir))
            
        try:
                print('Loading P data:')
                unstructured_grid_data_P = load_poly_data(f_name_P)
                pts_P = vtk_to_numpy(unstructured_grid_data_P.GetPoints().GetData()).astype(np.float32)
                np.save(save_path_srf + '/' +  'pts_P.npy', pts_P)
                print('pts_P.shape',pts_P.shape)
                
                edges_P = get_edges(unstructured_grid_data_P, pts_P)
                np.save(save_path_srf + '/' +  'edges_P.npy', edges_P)
                
                print('Surface calcs:')
                sdf_P = np.zeros(pts_P.shape[0]).astype(np.float32)
                normal_P = get_normal(unstructured_grid_data_P).astype(np.float32)
                surface = {tuple(p) for p in pts_P}
                
                print('Loading U data:')
                unstructured_grid_data_U = load_unstructured_grid_data(f_name_U)
                pts_U = vtk_to_numpy(unstructured_grid_data_U.GetPoints().GetData()).astype(np.float32)
                np.save(save_path_srf + '/' +  'pts_U.npy', pts_U)
                print('pts_U.shape',pts_U.shape)

                print('Volume calcs:')
                mesh = trimesh.load(geo_dir + '/' + b_name + '.obj', process=False)
                vert_up = mesh.vertices
                vert_up = vert_up[:,[0,2,1]]
                vert_up[:,0] = vert_up[:,0]*-1
                srf_mesh = trimesh.Trimesh(vertices=vert_up, faces=mesh.faces, process=False)
                sdf_U, normal_U = get_sdf_from_mesh(pts_U, srf_mesh)
                exterior_indices = [i for i, p in enumerate(pts_U) if tuple(p) not in surface]

                print('Combining points:')
                pos_ext = pts_U[exterior_indices]
                pos_surf = pts_P
                pos = np.concatenate([pos_ext, pos_surf])
                np.save(save_path_srf + '/' +  'pos.npy', pos)
                print(pos.shape)
                
                surf = np.concatenate([np.zeros(len(pos_ext)), np.ones(len(pos_surf))])
                np.save(save_path_srf + '/' + 'surf.npy', surf)
                print(surf.shape)

                print('Other combining calcs:')
                sdf_ext = sdf_U[exterior_indices]
                sdf_surf = sdf_P
                normal_ext = normal_U[exterior_indices]
                normal_surf = normal_P

                init_ext = np.c_[pos_ext, sdf_ext, normal_ext]
                init_surf = np.c_[pos_surf, sdf_surf, normal_surf]
                init = np.concatenate([init_ext, init_surf])
                   
                np.save(save_path_srf + '/' + 'x.npy', init)
                print(init.shape)
                   
        except Exception as e:
                print(f"Skipping {b} orientation {orient} due to error: {e}")
                continue  
    
    for orient in (np.arange(8)*45):

        b_dir = orient
        f_name_U = vtk_dir + '/' + b_name + '/' + b_name + '_' + str(b_dir) + '_U.vtk'   # Update your folder structure and directories
        f_name_P = vtk_dir + '/' + b_name + '/' + b_name + '_' + str(b_dir) + '_P.vtk'   # Update your folder structure and directories
        save_path_byO = vtknpy_dir + '/' + b_name + '/' + b_name + '_' + str(b_dir)      # Update your folder structure and directories
        
        if(os.path.exists(f_name_U) and os.path.exists(f_name_P) and (not all([os.path.exists(os.path.join(save_path_byO, nn+'.npy')) for nn in ['y']]))
          and (all([os.path.exists(os.path.join(save_path_srf, nn+'.npy')) for nn in ['x','pos','surf']]))):
            
            print('b_name',b_name)
            print('b_dir',b_dir)
            print('f_name_U',f_name_U)
            print('f_name_P',f_name_P)
            print('save_path_byO',save_path_byO)
            
            if (not os.path.exists(vtknpy_dir + '/' + b_name + '/' + b_name + '_' + str(b_dir))):
                os.makedirs(vtknpy_dir + '/' + b_name + '/' + b_name + '_' + str(b_dir))
                
            try:
                print('Loading P data:')
                unstructured_grid_data_P = load_poly_data(f_name_P)
                pts_P = np.load(save_path_srf + '/' +  'pts_P.npy')
                print('pts_P.shape',pts_P.shape)
                P = vtk_to_numpy(unstructured_grid_data_P.GetPointData().GetArray('p')).astype(np.float32)
                print('P.shape',P.shape)
                P_c = vtk_to_numpy(unstructured_grid_data_P.GetPointData().GetArray('total(p)_coeff')).astype(np.float32)
                print('P_c.shape',P_c.shape)

                print('Loading U data:')
                unstructured_grid_data_U = load_unstructured_grid_data(f_name_U)
                pts_U = np.load(save_path_srf + '/' +  'pts_U.npy')
                print('pts_U.shape',pts_U.shape)
                U = vtk_to_numpy(unstructured_grid_data_U.GetPointData().GetArray('U')).astype(np.float32)
                print('U.shape',U.shape)
                
                surface = {tuple(p) for p in pts_P}
                exterior_indices = [i for i, p in enumerate(pts_U) if tuple(p) not in surface]
                
                U_dict = {tuple(p): U[i] for i, p in enumerate(pts_U)}
                U_surf = np.array([U_dict[tuple(p)] if tuple(p) in U_dict else np.zeros(3) for p in pts_P])

                U_ext = U[exterior_indices]
                P_ext = np.zeros([len(exterior_indices), 2])
                   
                target_ext = np.c_[U_ext, P_ext]
                   
                P_surf = np.concatenate([np.expand_dims(P,0),np.expand_dims(P_c,0)],0).T
                
                target_surf = np.c_[U_surf, P_surf]

                target = np.concatenate([target_ext, target_surf])

                np.save(save_path_byO + '/' + 'y.npy', target)
                print(target.shape)

            except Exception as e:
                print(f"Skipping {b} orientation {orient} due to error: {e}")
                continue  

            
   