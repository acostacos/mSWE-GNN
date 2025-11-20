import numpy as np
import h5py
import geopandas as gpd
from dataclasses import dataclass
from shapely.geometry import Point, LineString

def extract_ghost_nodes(hecras_path: str, nodes_gdf: gpd.GeoDataFrame) -> np.ndarray:
    with h5py.File(hecras_path, 'r') as hec:
        min_elevation = np.array(hec['Geometry']['2D Flow Areas']['Perimeter 1']['Cells Minimum Elevation'])
    ghost_cells_idx = np.where(np.isnan(min_elevation))[0]
    for cell_idx in ghost_cells_idx:
        assert cell_idx in nodes_gdf.index, f"Ghost cell index {cell_idx} not found in nodes GeoDataFrame."
    return ghost_cells_idx

@dataclass
class HECRASMeshData:
    node_x: np.ndarray                # x-coordinates of mesh vertices
    node_y: np.ndarray                # y-coordinates of mesh vertices
    edge_index: np.ndarray            # connectivity of edges between vertices
    edge_type: np.ndarray             # type of each edge (1: normal, 2: boundary condition edge, 3: other boundary edges)

    face_x: np.ndarray                # x-coordinates of face centers
    face_y: np.ndarray                # y-coordinates of face centers
    dual_edge_index: np.ndarray       # connectivity of dual edges between face centers
    face_nodes: np.ndarray = None     # connectivity of faces to their vertices

    @classmethod
    def from_gdfs(cls,
                  nodes_gdf: gpd.GeoDataFrame,
                  edges_gdf: gpd.GeoDataFrame,
                  faces_gdf: gpd.GeoDataFrame,
                  facepoints_gdf: gpd.GeoDataFrame,
                  ghost_nodes: np.ndarray = None,
                  inflow_bc_nodes: np.ndarray = None):
        """Initialize HECRASMeshData from GeoDataFrames.

        Args:
            nodes_gdf (gpd.GeoDataFrame): GeoDataFrame containing mesh vertices.
                columns: 'X', 'Y', 'CC_index', 'geometry'
            edges_gdf (gpd.GeoDataFrame): GeoDataFrame containing mesh edges.
                columns: 'link_index', 'from_node', 'to_node', 'geometry'
            faces_gdf (gpd.GeoDataFrame): GeoDataFrame containing mesh faces.
                columns: 'face_index', 'from', 'to', 'geometry'
            facepoints_gdf (gpd.GeoDataFrame): GeoDataFrame containing face center points.
                columns: 'X', 'Y', 'FP_index', 'geometry'
            ghost_nodes (np.ndarray, optional): Indices of ghost nodes to exclude. Defaults to None.
            inflow_bc_nodes (np.ndarray, optional): Indices of inflow boundary condition nodes. Defaults to None.
        """
        if ghost_nodes is not None:
            if inflow_bc_nodes is not None:
                assert inflow_bc_nodes in ghost_nodes, "Inflow boundary condition nodes must be a subset of ghost nodes."

            ghost_nodes_gdf = nodes_gdf.iloc[ghost_nodes]
            nodes_gdf = nodes_gdf.drop(index=ghost_nodes)

            ghost_edges_mask = (edges_gdf['from_node'].isin(ghost_nodes) | edges_gdf['to_node'].isin(ghost_nodes))
            ghost_edges_gdf = edges_gdf[ghost_edges_mask]
            edges_gdf = edges_gdf[~ghost_edges_mask]

        node_x = nodes_gdf['X'].to_numpy()
        node_y = nodes_gdf['Y'].to_numpy()
        edge_index = faces_gdf[['from', 'to']].to_numpy().T

        # Create edge_type (default to all normal edges for now)
        edge_type = np.ones(edge_index.shape[1], dtype=int)

        face_x = facepoints_gdf['X'].to_numpy()
        face_y = facepoints_gdf['Y'].to_numpy()
        dual_edge_index = edges_gdf[['from_node', 'to_node']].to_numpy().T

        # TODO: Implement
        face_nodes = None

        return cls(
            node_x=node_x,
            node_y=node_y,
            edge_index=edge_index,
            edge_type=edge_type,
            face_x=face_x,
            face_y=face_y,
            dual_edge_index=dual_edge_index,
            face_nodes=face_nodes,
        )

    def to_nodes_gdf(self) -> gpd.GeoDataFrame:
        """
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with node geometry and attributes
                columns: 'X', 'Y', 'CC_index', 'geometry'
        """
        nodes_geom = [Point(x, y) for x, y in zip(self.node_x, self.node_y)]
        nodes_gdf = gpd.GeoDataFrame({
            'X': self.node_x,
            'Y': self.node_y,
            'CC_index': np.arange(len(self.node_x))
        }, geometry=nodes_geom)
        return nodes_gdf
    
    def to_edges_gdf(self) -> gpd.GeoDataFrame:
        """
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with edge geometry and attributes
                columns: 'link_index', 'from_node', 'to_node', 'edge_type', 'geometry'
        """
        edges_list = []
        for i in range(self.dual_edge_index.shape[1]):
            from_node = self.dual_edge_index[0, i]
            to_node = self.dual_edge_index[1, i]
            edge_geom = LineString([
                (self.node_x[from_node], self.node_y[from_node]),
                (self.node_x[to_node], self.node_y[to_node])
            ])
            edges_list.append({
                'link_index': i,
                'from_node': from_node,
                'to_node': to_node,
                'edge_type': self.edge_type[i],
                'geometry': edge_geom
            })
        edges_gdf = gpd.GeoDataFrame(edges_list)
        return edges_gdf

    def to_facepoints_gdf(self) -> gpd.GeoDataFrame:
        """
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with face point geometry and attributes
                columns: 'X', 'Y', 'FP_index', 'geometry'
        """
        fp_geom = [Point(x, y) for x, y in zip(self.face_x, self.face_y)]
        fp_gdf = gpd.GeoDataFrame({
            'X': self.face_x,
            'Y': self.face_y,
            'FP_index': np.arange(len(self.face_x))
        }, geometry=fp_geom)
        return fp_gdf
    
    def to_faces_gdf(self) -> gpd.GeoDataFrame:
        """
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with face edge geometry and attributes
                columns: 'face_index', 'from', 'to', 'geometry'
        """
        faces_list = []
        for i in range(self.edge_index.shape[1]):
            from_idx = self.edge_index[0, i]
            to_idx = self.edge_index[1, i]
            if from_idx >= 0 and to_idx >= 0:
                edge_geom = LineString([
                    (self.face_x[from_idx], self.face_y[from_idx]),
                    (self.face_x[to_idx], self.face_y[to_idx])
                ])
                faces_list.append({
                    'face_index': i,
                    'from': from_idx,
                    'to': to_idx,
                    'geometry': edge_geom
                })
        faces_gdf = gpd.GeoDataFrame(faces_list)
        return faces_gdf

