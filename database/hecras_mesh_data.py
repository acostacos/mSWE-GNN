import numpy as np
import geopandas as gpd
from dataclasses import dataclass

from collections import OrderedDict
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from typing import Tuple, Optional

@dataclass
class HECRASMeshData:
    # Mesh data for mapping
    # Directly from HECRAS and shapefiles
    face_x: np.ndarray                              # x-coordinates of face centers (nodes in graph)
    face_y: np.ndarray                              # y-coordinates of face centers (nodes in graph)
    dual_edge_index: np.ndarray                     # connectivity of dual edges between face centers (edges in graph)

    # Derived mesh attributes
    node_x: np.ndarray                              # x-coordinates of mesh vertices
    node_y: np.ndarray                              # y-coordinates of mesh vertices
    edge_index: np.ndarray                          # connectivity of edges between vertices
    edge_type: np.ndarray                           # type of each edge in edge_index (1: normal, 2: boundary condition edge, 3: other boundary edges)
    face_nodes: np.ndarray                          # List of nodes (vertices) that make up each face

    # HECRAS specific data
    inflow_bc_gdf: gpd.GeoDataFrame = None          # GeoDataFrame of inflow boundary condition nodes. Contains x and y coordinates of inflow BC nodes from HECRAS.

    @classmethod
    def from_gdfs(cls,
                  nodes_gdf: gpd.GeoDataFrame,
                  edges_gdf: gpd.GeoDataFrame,
                  faces_gdf: gpd.GeoDataFrame,
                  ghost_nodes: np.ndarray = None,
                  inflow_bc_nodes: np.ndarray = None):
        """Initialize HECRASMeshData from GeoDataFrames. Typically used when initially reading from raw HECRAS and shapefiles.

        Args:
            nodes_gdf (gpd.GeoDataFrame): GeoDataFrame containing mesh vertices.
                columns: 'X', 'Y', 'CC_index', 'geometry'
            edges_gdf (gpd.GeoDataFrame): GeoDataFrame containing mesh edges.
                columns: 'link_index', 'from_node', 'to_node', 'geometry'
            faces_gdf (gpd.GeoDataFrame): GeoDataFrame containing mesh faces.
                columns: 'face_index', 'from', 'to', 'geometry'
            ghost_nodes (np.ndarray, optional): Indices of ghost nodes to exclude. Defaults to None.
            inflow_bc_nodes (np.ndarray, optional): Indices of inflow boundary condition nodes. Defaults to None.
        """
        if ghost_nodes is not None:
            # Only remove ghost nodes from nodes_gdf and edges_gdf; They do not affect faces_gdf
            ghost_nodes_mask = nodes_gdf['CC_index'].isin(ghost_nodes)
            ghost_nodes_gdf = nodes_gdf[ghost_nodes_mask]
            nodes_gdf = nodes_gdf[~ghost_nodes_mask]

            def flip_edge_direction(gdf, to_flip_mask):
                gdf.loc[to_flip_mask, ['from_node', 'to_node']] = gdf.loc[to_flip_mask, ['to_node', 'from_node']].values

            if inflow_bc_nodes is not None:
                assert inflow_bc_nodes in ghost_nodes, "Inflow boundary condition nodes must be a subset of ghost nodes."

                inflow_bc_gdf = ghost_nodes_gdf[ghost_nodes_gdf['CC_index'].isin(inflow_bc_nodes)]

                inflow_to_flip_mask = edges_gdf['to_node'].isin(inflow_bc_nodes)
                flip_edge_direction(edges_gdf, inflow_to_flip_mask)
                inflow_edges_mask = edges_gdf['from_node'].isin(inflow_bc_nodes)
                edges_gdf.loc[inflow_edges_mask, 'from_node'] = -1

            ghost_to_flip_mask = edges_gdf['from_node'].isin(ghost_nodes)
            flip_edge_direction(edges_gdf, ghost_to_flip_mask)
            ghost_edges_mask = edges_gdf['to_node'].isin(ghost_nodes)
            edges_gdf.loc[ghost_edges_mask, 'to_node'] = -1

        face_x = nodes_gdf['X'].to_numpy()
        face_y = nodes_gdf['Y'].to_numpy()
        dual_edge_index = edges_gdf[['from_node', 'to_node']].to_numpy().T

        return cls(
            face_x=face_x,
            face_y=face_y,
            dual_edge_index=dual_edge_index,
            faces_gdf=faces_gdf,
            inflow_bc_gdf=inflow_bc_gdf if inflow_bc_nodes is not None else None
        )

    def __init__(self,
                 face_x: np.ndarray,
                 face_y: np.ndarray,
                 dual_edge_index: np.ndarray,
                 faces_gdf: gpd.GeoDataFrame,
                 inflow_bc_gdf: gpd.GeoDataFrame = None):
        self.face_x = face_x
        self.face_y = face_y
        self.dual_edge_index = dual_edge_index
        self.inflow_bc_gdf = inflow_bc_gdf
        self._get_derived_attributes(faces_gdf, inflow_bc_gdf)

    def _get_derived_attributes(self, faces_gdf: gpd.GeoDataFrame, inflow_bc_gdf: Optional[gpd.GeoDataFrame]):
        self.edge_index, self.edge_type = self._get_edge_attributes(faces_gdf, inflow_bc_gdf)
        self.node_x, self.node_y, self.face_nodes = self._get_node_attributes(faces_gdf, self.edge_index)

    def _get_edge_attributes(self,
                             faces_gdf: gpd.GeoDataFrame,
                             inflow_bc_gdf: Optional[gpd.GeoDataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        '''Create mesh node/vertex attributes: edge_index, edge_type'''
        edge_index = faces_gdf[['from', 'to']].to_numpy().T

        merged_polygon = unary_union(faces_gdf.polygonize().to_list())
        boundary_shape = merged_polygon.exterior
        edge_type = np.ones(edge_index.shape[1], dtype=int) # Default = edge type 1

        # Filter for edges closest to the boundary
        boundary_edge_matches = faces_gdf.sindex.query(boundary_shape, predicate='intersects')

        # Check which of these actually correspond to boundary edges
        boundary_coords = list(merged_polygon.exterior.coords)
        boundary_lines = [LineString([boundary_coords[i], boundary_coords[i+1]]) for i in range(len(boundary_coords)-1)]
        for edge_idx in boundary_edge_matches:
            edge_shape = faces_gdf.geometry.iloc[edge_idx]
            for line in boundary_lines:
                if edge_shape.equals(line):
                    if inflow_bc_gdf is not None:
                        edge_buffer = edge_shape.buffer(1e-6)  # Small buffer for point-line intersection
                        inflow_bc_edge_idxs = inflow_bc_gdf.sindex.query(edge_buffer, predicate='intersects')
                        if len(inflow_bc_edge_idxs) > 0:
                            edge_type[edge_idx] = 2  # Inflow BC edges = edge type 2
                            continue
                    edge_type[edge_idx] = 3  # Boundary edges = edge type 3

        return edge_index, edge_type
 
    def _get_node_attributes(self, faces_gdf: gpd.GeoDataFrame, edge_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''Create mesh node/vertex attributes: node_x, node_y, face_nodes'''
        nodes_gdf = self.to_nodes_gdf()
        mesh_faces = faces_gdf.polygonize()
        assert len(mesh_faces) == len(nodes_gdf), "Number of mesh faces must equal number of face centers (nodes)."

        # For node_x and node_y
        node_x = np.full(edge_index.max() + 1, np.nan)
        node_y = np.full(edge_index.max() + 1, np.nan)

        # For face_nodes
        face_nodes_list = [[] for _ in range(len(nodes_gdf))]

        for face_idx, face in enumerate(mesh_faces):
            assert face.geom_type == 'Polygon', f"Face {face_idx} is not a Polygon even if all faces are expected to be closed."

            # Get respective face center index (node in graph)
            face_center_matches = nodes_gdf.sindex.query(face, predicate='intersects')
            assert len(face_center_matches) > 0, f"Face center not found for face {face_idx}."
            if len(face_center_matches) > 1:
                print(f"Warning : Multiple face centers found for face {face_idx} - {len(face_center_matches)} candidates.")
            face_center_idx = face_center_matches[0]

            face_vertices = list(face.exterior.coords)[:-1]  # Exclude last duplicate point
            vertex_indices = OrderedDict()
            for vertex_idx in range(len(face_vertices) - 1):
                face_edge_coords = [face_vertices[vertex_idx], face_vertices[vertex_idx + 1]]

                # Check which index of edge_index this edge corresponds to
                edge_matches = faces_gdf.sindex.query(LineString(face_edge_coords), predicate='intersects')

                # Filter to edges that are actually part of the polygon boundary
                match_found = False
                for edge_idx in edge_matches:
                    edges_idxs = edge_index[:, edge_idx]
                    edge_coords = list(faces_gdf.geometry.iloc[edge_idx].coords)
                    if edge_coords == face_edge_coords[::-1]:
                        edge_coords = edge_coords[::-1]
                        edges_idxs = edges_idxs[::-1]
                    if edge_coords == face_edge_coords:
                        match_found = True
                        break
                assert match_found, f"No matching edge found for face {face_idx} between vertices {vertex_idx} and {vertex_idx+1}."

                for coord_idx, coords in enumerate(face_edge_coords):
                    node_idx = edges_idxs[coord_idx]
                    if node_idx not in vertex_indices:
                        vertex_indices[node_idx] = None

                        node_x[node_idx] = coords[0]
                        node_y[node_idx] = coords[1]
            face_nodes_list[face_center_idx] = list(vertex_indices.keys())

        assert all(len(face) > 0 for face in face_nodes_list), "Some faces have no associated nodes/vertices."
        assert all(not np.isnan(x) for x in node_x), "Some node_x values are NaN as they have not been assigned."
        assert all(not np.isnan(y) for y in node_y), "Some node_y values are NaN as they have not been assigned."

        # Convert to padded numpy array with NaN for different vertex counts
        max_vertices = max(len(face) for face in face_nodes_list)
        face_nodes = np.full((len(face_nodes_list), max_vertices), np.nan)
        for vertex_idx, vertex_indices in enumerate(face_nodes_list):
            face_nodes[vertex_idx, :len(vertex_indices)] = vertex_indices

        return node_x, node_y, face_nodes

    def to_nodes_gdf(self) -> gpd.GeoDataFrame:
        """
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with node geometry and attributes
                columns: 'X', 'Y', 'CC_index', 'geometry'
        """
        nodes_geom = [Point(x, y) for x, y in zip(self.face_x, self.face_y)]
        nodes_gdf = gpd.GeoDataFrame({
            'X': self.face_x,
            'Y': self.face_y,
            'CC_index': np.arange(len(self.face_x))
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
                (self.face_x[from_node], self.face_y[from_node]),
                (self.face_x[to_node], self.face_y[to_node])
            ])
            edges_list.append({
                'link_index': i,
                'from_node': from_node,
                'to_node': to_node,
                'geometry': edge_geom
            })
        edges_gdf = gpd.GeoDataFrame(edges_list)
        return edges_gdf

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
                    (self.node_x[from_idx], self.node_y[from_idx]),
                    (self.node_x[to_idx], self.node_y[to_idx])
                ])
                faces_list.append({
                    'face_index': i,
                    'from': from_idx,
                    'to': to_idx,
                    'edge_type': self.edge_type[i],
                    'geometry': edge_geom
                })
        faces_gdf = gpd.GeoDataFrame(faces_list)
        return faces_gdf
