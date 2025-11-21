import os
import yaml
import pandas as pd
import torch
import numpy as np
import geopandas as gpd

from copy import copy
from graph_creation import create_dataset_folders, save_database, create_hecras_multiscale_mesh, \
    interpolate_BC_location_multiscale, add_ghost_cells_mesh, add_ghost_cells_attributes, \
    pool_multiscale_attributes, update_ghost_cells_attributes, MultiscaleMesh
from torch_geometric.data import Data
from transform_helper_files.hecras_data_retrieval import get_water_level, get_velocity, get_face_flow, \
    get_cell_area, get_facepoint_coordinates, get_edge_direction_x, get_edge_direction_y, \
    get_face_length, get_facecell_indexes, get_facepoint_indexes
from transform_helper_files.shp_data_retrieval import get_cell_elevation, get_edge_index, get_cell_position
from hecras_mesh_data import HECRASMeshData, extract_ghost_nodes

def get_info_from_config(config_file_path: str, root_dir: str, faces_shp_file: str, facepoints_shp_file) -> dict:
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    dataset_config = config['dataset_parameters']
    nodes_shp_path = os.path.join(root_dir, 'raw', dataset_config['nodes_shp_file'])
    edges_shp_path = os.path.join(root_dir, 'raw', dataset_config['edges_shp_file'])
    faces_shp_path = os.path.join(root_dir, 'raw', faces_shp_file)
    facepoints_shp_path = os.path.join(root_dir, 'raw', facepoints_shp_file)
    train_summary_path = os.path.join(root_dir, 'raw', dataset_config['training']['dataset_summary_file'])
    test_summary_path = os.path.join(root_dir, 'raw', dataset_config['testing']['dataset_summary_file'])
    inflow_boundary_nodes = dataset_config['inflow_boundary_nodes']

    return {
        'nodes_shp_path': nodes_shp_path,
        'edges_shp_path': edges_shp_path,
        'faces_shp_path': faces_shp_path,
        'facepoints_shp_path': facepoints_shp_path,
        'train_summary_path': train_summary_path,
        'test_summary_path': test_summary_path,
        'inflow_boundary_nodes': inflow_boundary_nodes,
    }

def get_dataset_info_from_summary(summary_path: str,
                                  root_dir: str,
                                  nodes_shp_path: str,
                                  edges_shp_path: str,
                                  faces_shp_path: str,
                                  facepoints_shp_path: str,
                                  inflow_boundary_nodes: list[int]) -> dict:
    summary_df = pd.read_csv(summary_path)

    datasets = {}
    for _, row in summary_df.iterrows():
        run_id = row['Run_ID']
        hec_ras_path = row['HECRAS_Filepath']
        datasets[run_id] = {
            'hec_ras_file_path': os.path.join(root_dir, 'raw', hec_ras_path),
            'node_shp_path': nodes_shp_path,
            'edge_shp_path': edges_shp_path,
            'face_shp_path': faces_shp_path,
            'facepoint_shp_path': facepoints_shp_path,
            'inflow_boundary_nodes': inflow_boundary_nodes,
        }
    return datasets

def get_cell_velocity(hec_ras_filepath: str, node_shp_filepath: str, perimeter_name: str = 'Perimeter 1') -> torch.Tensor:
    '''Adopted from https://doi.org/10.26188/24312658'''
    def dist_center2faces(center_xy,faces_xy):
        dist = np.sqrt(np.square(faces_xy[:,0]-center_xy[0]) + np.square(faces_xy[:,1]-center_xy[1]))
        return dist

    xy_coor = get_cell_position(node_shp_filepath)
    cell_area = get_cell_area(hec_ras_filepath, perimeter_name)
    facepoint_xy_coor = get_facepoint_coordinates(hec_ras_filepath, perimeter_name)
    edge_direction_x = get_edge_direction_x(hec_ras_filepath, perimeter_name)
    edge_direction_y = get_edge_direction_y(hec_ras_filepath, perimeter_name)
    face_length = get_face_length(hec_ras_filepath, perimeter_name)
    faces_cell_idx = get_facecell_indexes(hec_ras_filepath, perimeter_name)
    faces_facepoint_idx = get_facepoint_indexes(hec_ras_filepath, perimeter_name)
    face_vel = get_velocity(hec_ras_filepath, perimeter_name)

    # Find x-y components for each cell face in each cell
    n_timesteps = len(face_vel)
    n_cells = len(xy_coor)

    cell_velocity_x = np.zeros([n_timesteps, n_cells])
    cell_velocity_y = np.zeros([n_timesteps, n_cells])
    for cell_i in range(n_cells):
        # Find cell in FROM/TO table of faces
        find_faces_for_cell = np.column_stack(np.where(faces_cell_idx == cell_i))

        # Find cell velocity x-y components each cell
        # HEC-RAS method: https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/latest/theoretical-basis-for-one-dimensional-and-two-dimensional-hydrodynamic-calculations/2d-unsteady-flow-hydrodynamics/numerical-methods/cell-velocity
        # Vc = 1/A * SUM(dx * L * v_f)
        # where A is the cell area, 
        # dx is distance from cell center to facecenter, 
        # L is face length, v_f is facevelocity
        cell_xy = xy_coor[cell_i]
        facepoint1_xy = facepoint_xy_coor[faces_facepoint_idx[find_faces_for_cell[:,0]][:,0]]
        facepoint2_xy = facepoint_xy_coor[faces_facepoint_idx[find_faces_for_cell[:,0]][:,1]]
        face_center_xy = np.c_[np.mean([facepoint1_xy[:,0], facepoint2_xy[:,0]], axis=0),
                    np.mean([facepoint1_xy[:,1], facepoint2_xy[:,1]], axis=0)]

        dx_center2face = dist_center2faces(cell_xy,face_center_xy)

        if cell_area[cell_i] == 0 or np.isnan(cell_area[cell_i]):
            continue
        else:
            cell_velocity_x[:,cell_i] = 1/cell_area[cell_i] * np.sum(dx_center2face * 
                                                face_length[find_faces_for_cell[:,0]] * 
                                                face_vel[:, find_faces_for_cell[:,0]] * edge_direction_x[find_faces_for_cell[:,0]], axis=1)

            cell_velocity_y[:,cell_i] = 1/cell_area[cell_i] * np.sum(dx_center2face *
                                                    face_length[find_faces_for_cell[:,0]] *
                                                    face_vel[:, find_faces_for_cell[:,0]] * edge_direction_y[find_faces_for_cell[:,0]], axis=1)

    return torch.FloatTensor(cell_velocity_x), torch.FloatTensor(cell_velocity_y)

def get_inflow(hec_ras_path: str, edges_shp_path: str, inflow_boundary_nodes: list[int]):
    """Get inflow at boundary nodes"""
    face_flow = get_face_flow(hec_ras_path)
    edge_index = get_edge_index(edges_shp_path)
    inflow_to_boundary_mask = np.isin(edge_index[1], inflow_boundary_nodes)
    if np.any(inflow_to_boundary_mask):
        # Flip the dynamic edge features accordingly
        face_flow[:, inflow_to_boundary_mask] *= -1

    inflow_edges_mask = np.any(np.isin(edge_index, inflow_boundary_nodes), axis=0)
    inflow = face_flow[:, inflow_edges_mask].sum(axis=1)

    return inflow

def get_hydraulic_features(hec_ras_file_path: str,
                           node_shp_path: str,
                           edges_shp_path: str,
                           inflow_boundary_nodes: list[int],
                           spin_up_timesteps: int = None,
                           ts_from_peak_water_depth: int = None,
                           downsample_interval: int = None):
    cell_velocity_x, cell_velocity_y = get_cell_velocity(hec_ras_file_path, node_shp_path)
    dem = torch.FloatTensor(get_cell_elevation(node_shp_path))
    water_level = torch.FloatTensor(get_water_level(hec_ras_file_path))
    water_depth = torch.clip(water_level - dem, min=0)
    inflow = get_inflow(hec_ras_file_path, edges_shp_path, inflow_boundary_nodes)

    # TODO: Implement spin-up, ts_from_peak_water_depth, downsample_interval

    return dem, water_depth, cell_velocity_x, cell_velocity_y, inflow

def create_mesh_data_from_files(hec_ras_file_path: str,
                                node_shp_path: str,
                                edge_shp_path: str,
                                face_shp_path: str,
                                facepoint_shp_path: str,
                                inflow_boundary_nodes: list[int]):
    face_centers_gdf = gpd.read_file(node_shp_path)
    dual_edges_gdf = gpd.read_file(edge_shp_path)
    face_edges_gdf = gpd.read_file(face_shp_path)
    face_vertices_gdf = gpd.read_file(facepoint_shp_path)
    ghost_nodes = extract_ghost_nodes(hec_ras_file_path, face_centers_gdf)

    mesh_data = HECRASMeshData.from_gdfs(
        face_centers_gdf,
        dual_edges_gdf,
        face_edges_gdf,
        face_vertices_gdf,
        ghost_nodes,
        inflow_boundary_nodes,
    )
    return mesh_data

def create_mesh_dataset(dataset_info: dict,
                        spin_up_timesteps: int = None,
                        ts_from_peak_water_depth: int = None,
                        downsample_interval: int = None,
                        number_of_multiscales=4):
    mesh_dataset = []

    for key, dataset_info in dataset_info.items():
        print(f"Processing event {key}", flush=True)
        hec_ras_file_path = dataset_info['hec_ras_file_path']
        node_shp_path = dataset_info['node_shp_path']
        edge_shp_path = dataset_info['edge_shp_path']
        face_shp_path = dataset_info['face_shp_path']
        facepoint_shp_path = dataset_info['facepoint_shp_path']
        inflow_boundary_nodes = dataset_info['inflow_boundary_nodes']

        data = convert_mesh_to_pyg(hec_ras_file_path, node_shp_path, edge_shp_path, face_shp_path,
                                   facepoint_shp_path, inflow_boundary_nodes, spin_up_timesteps,
                                   ts_from_peak_water_depth, downsample_interval, number_of_multiscales)
        mesh_dataset.append(data)
    
    return mesh_dataset

def convert_mesh_to_pyg(hec_ras_file_path: str,
                        node_shp_path: str,
                        edge_shp_path: str,
                        face_shp_path: str,
                        facepoint_shp_path: str,
                        inflow_boundary_nodes: list[int],
                        spin_up_timesteps: int = None,
                        ts_from_peak_water_depth: int = None,
                        downsample_interval: int = None,
                        number_of_multiscales: int = 4):
    DEM, WD, VX, VY, BC = get_hydraulic_features(hec_ras_file_path,
                                                 node_shp_path,
                                                 edge_shp_path,
                                                 inflow_boundary_nodes,
                                                 spin_up_timesteps,
                                                 ts_from_peak_water_depth,
                                                 downsample_interval)
    # BC[:,0] /= 60 # convert to minutes # TODO: See if you need this

    data = Data()

    mesh_data = create_mesh_data_from_files(hec_ras_file_path,
                                            node_shp_path,
                                            edge_shp_path,
                                            face_shp_path,
                                            facepoint_shp_path,
                                            inflow_boundary_nodes)

    # create multiscale meshes
    meshes = create_hecras_multiscale_mesh(mesh_data=mesh_data,
                                           coarsening_factor=0.5,
                                           number_of_multiscales=number_of_multiscales-1)
    meshes.append(copy(meshes[0]))
    meshes[-1]._import_from_hecras_data(mesh_data)
    # TODO: See if you need to reverse normals
    # meshes[-1].edge_outward_normal[meshes[-1].edge_BC] *= -1  # reverse the normal of the boundary edges
    meshes = meshes[::-1]

    # Add boundary conditions to multiscale meshes
    edge_BC_mid = mesh.node_xy[mesh.edge_index_BC].mean(1)
    meshes = interpolate_BC_location_multiscale(meshes, edge_BC_mid)
    meshes = [add_ghost_cells_mesh(mesh) for mesh in meshes]
    DEM, WD, VX, VY = add_ghost_cells_attributes(meshes[0], DEM, WD, VX, VY)

    # create multiscale mesh
    mesh = MultiscaleMesh()
    mesh.stack_meshes(meshes)

    data.node_ptr = torch.LongTensor(mesh.face_ptr)
    data.edge_ptr = torch.LongTensor(mesh.dual_edge_ptr)
    data.intra_edge_ptr = torch.LongTensor(mesh.intra_edge_ptr)
    data.intra_mesh_edge_index = torch.LongTensor(mesh.intra_mesh_dual_edge_index)

    # get multiscale attributes
    # mesh.DEM, WD, VX, VY = interpolate_multiscale_attributes(meshes, DEM, WD, VX, VY, method='nearest')
    mesh.DEM, WD, VX, VY = pool_multiscale_attributes(mesh, DEM, WD, VX, VY, reduce='mean')
    mesh.DEM = update_ghost_cells_attributes(mesh, mesh.DEM)[0] #correct ghost cells values after pooling







    # Assign data features
    data.DEM = torch.FloatTensor(mesh.DEM)
    data.WD = torch.FloatTensor(WD)
    data.VX = torch.FloatTensor(VX)
    data.VY = torch.FloatTensor(VY)
    # data.slopex = torch.FloatTensor(slope_x)
    # data.slopey = torch.FloatTensor(slope_y)
    
    # Assign other data properties
    data.edge_index = torch.LongTensor(mesh.dual_edge_index)
    data.face_distance = torch.FloatTensor(mesh.dual_edge_length)
    data.face_relative_distance = torch.FloatTensor(mesh.face_relative_distance)
    data.edge_slope = (data.DEM[data.edge_index][0] - data.DEM[data.edge_index][1])/data.face_distance
    # data.normal = torch.FloatTensor(mesh.edge_outward_normal[mesh.edge_type < 3])
    data.num_nodes = mesh.face_x.shape[0]
    data.area = torch.FloatTensor(mesh.face_area)

    data.mesh = mesh
    
    data.node_BC = data.node_BC[:len(mesh.ghost_cells_ids)//number_of_multiscales] # select BC only at the finest scale
    data.edge_BC_length = data.edge_BC_length[:len(mesh.ghost_cells_ids)//number_of_multiscales] # select BC+edge only at the finest scale
    data.BC = torch.FloatTensor(BC).unsqueeze(0).repeat(len(data.node_BC), 1, 1) # This repeats the same BC
    data.type_BC = torch.tensor(2, dtype=torch.int) # 2 = inflow / discharge BC type

    return data

def main():
    root_dir = ""
    config_file_path = ""
    faces_shp_file = ""
    facepoints_shp_file = ""
    base_dataset_folder = ""
    spin_up_timesteps = 864
    ts_from_peak_water_depth = None # Set to None to disable
    downsample_interval = 3

    info = get_info_from_config(config_file_path, root_dir, faces_shp_file, facepoints_shp_file)
    create_dataset_folders(dataset_folder=base_dataset_folder)

    # Training dataset creation
    train_datasets = get_dataset_info_from_summary(info['train_summary_path'],
                                                   root_dir,
                                                   info['nodes_shp_path'],
                                                   info['edges_shp_path'],
                                                   info['faces_shp_path'],
                                                   info['facepoints_shp_path'],
                                                   info['inflow_boundary_nodes'])

    train_pyg_dataset = create_mesh_dataset(train_datasets,
                                            spin_up_timesteps,
                                            ts_from_peak_water_depth,
                                            downsample_interval)
    train_folder = f"{base_dataset_folder}/train"
    save_database(train_pyg_dataset, name='hecras', out_path=train_folder)
    print(f"Training dataset created and saved in folder {train_folder}.")

    # Testing dataset creation
    # test_datasets = get_dataset_info_from_summary(info['test_summary_path'],
    #                                               root_dir,
    #                                               info['nodes_shp_path'],
    #                                               info['edges_shp_path'],
    #                                               info['faces_shp_path'],
    #                                               info['facepoints_shp_path'],
    #                                               info['inflow_boundary_nodes'])

    # for key, paths in test_datasets.items():
    #     test_pyg_dataset = create_mesh_dataset({key: paths},
    #                                         spin_up_timesteps,
    #                                         ts_from_peak_water_depth,
    #                                         downsample_interval)
    #     test_folder = f"{base_dataset_folder}/test"
    #     save_database(test_pyg_dataset, name=key, out_path=test_folder)
    #     print(f"Testing dataset for Event {key} created and saved in folder {test_folder}.")

if __name__ == "__main__":
    main()