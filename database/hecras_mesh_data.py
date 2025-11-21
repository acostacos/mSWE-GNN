import numpy as np
import h5py
import geopandas as gpd
from dataclasses import dataclass
from shapely.geometry import Point, LineString
from shapely.ops import unary_union

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

    face_x: np.ndarray                # x-coordinates of face centers (nodes in graph)
    face_y: np.ndarray                # y-coordinates of face centers (nodes in graph)
    dual_edge_index: np.ndarray       # connectivity of dual edges between face centers (edges in graph)
    face_nodes: np.ndarray = None     # List of nodes (vertices) that make up each face

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
            ghost_nodes_gdf = nodes_gdf[nodes_gdf['CC_index'].isin(ghost_nodes)]
            nodes_gdf = nodes_gdf.drop(index=ghost_nodes)

            ghost_edges_mask = (edges_gdf['from_node'].isin(ghost_nodes) | edges_gdf['to_node'].isin(ghost_nodes))
            ghost_edges_gdf = edges_gdf[ghost_edges_mask]
            edges_gdf = edges_gdf[~ghost_edges_mask]

            if inflow_bc_nodes is not None:
                assert inflow_bc_nodes in ghost_nodes, "Inflow boundary condition nodes must be a subset of ghost nodes."

                inflow_nodes_gdf = ghost_nodes_gdf[ghost_nodes_gdf['CC_index'].isin(inflow_bc_nodes)]

        node_x = facepoints_gdf['X'].to_numpy()
        node_y = facepoints_gdf['Y'].to_numpy()
        edge_index = faces_gdf[['from', 'to']].to_numpy().T

        edge_type = HECRASMeshData.create_edge_types(
            edge_index=edge_index,
            node_x=node_x,
            node_y=node_y,
            faces_gdf=faces_gdf,
            inflow_nodes_gdf=inflow_nodes_gdf if inflow_bc_nodes is not None else None,
        )

        face_x = nodes_gdf['X'].to_numpy()
        face_y = nodes_gdf['Y'].to_numpy()
        dual_edge_index = edges_gdf[['from_node', 'to_node']].to_numpy().T

        face_nodes = HECRASMeshData.create_face_nodes(
            nodes_gdf=nodes_gdf,
            faces_gdf=faces_gdf,
            facepoints_gdf=facepoints_gdf,
        )

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

    @classmethod
    def create_edge_types(cls,
                          edge_index: np.ndarray,
                          node_x: np.ndarray,
                          node_y: np.ndarray,
                          faces_gdf: gpd.GeoDataFrame,
                          inflow_nodes_gdf: gpd.GeoDataFrame = None) -> np.ndarray:
        edge_type = np.ones(edge_index.shape[1], dtype=int) # Default = edge type 1
        mesh_cells = faces_gdf.polygonize()
        merged_polygon = unary_union(mesh_cells.to_list())
        boundary_coords = np.array(list(merged_polygon.exterior.coords))
        if inflow_nodes_gdf is not None:
            inflow_coords = inflow_nodes_gdf[['X', 'Y']].to_numpy()
            inflow_points = [Point(x, y) for x, y in inflow_coords]

        def is_on_boundary(px: float, py: float, tolerance: float = 1e-6) -> bool:
            distances = np.sqrt((boundary_coords[:, 0] - px)**2 + (boundary_coords[:, 1] - py)**2)
            return np.min(distances) < tolerance

        for i in range(edge_index.shape[1]):
            from_idx = edge_index[0, i]
            to_idx = edge_index[1, i]
            from_point_x, from_point_y = node_x[from_idx], node_y[from_idx]
            to_point_x, to_point_y = node_x[to_idx], node_y[to_idx]

            # Check if both endpoints are on the boundary
            if not (is_on_boundary(from_point_x, from_point_y) and is_on_boundary(to_point_x, to_point_y)):
                continue

            if inflow_nodes_gdf is not None:
                line = LineString([(from_point_x, from_point_y), (to_point_x, to_point_y)])
                found_inflow = False
                for inflow_point in inflow_points:
                    # Edges that are linked to face of inflow boundary condition = edge type 2
                    if line.distance(inflow_point) < 1e-6:
                        edge_type[i] = 2
                        found_inflow = True
                        break
                if found_inflow:
                    continue

            # Edges at boundary of the mesh = edge type 3
            edge_type[i] = 3
        return edge_type

    @classmethod
    def create_face_nodes(cls,
                          nodes_gdf: gpd.GeoDataFrame,
                          faces_gdf: gpd.GeoDataFrame,
                          facepoints_gdf: gpd.GeoDataFrame) -> np.ndarray:
        # Populate face_nodes: map each face center to its surrounding vertices
        # 1. Create mesh geometries from faces_gdf
        mesh_cells = faces_gdf.polygonize()
        spatial_index = mesh_cells.sindex
        face_center_coords = nodes_gdf[['X', 'Y']].to_numpy()

        # 2. Get facepoints coordinates as numpy array for efficient nearest neighbor search
        facepoints_coords = facepoints_gdf[['X', 'Y']].to_numpy()
        
        def find_nearest_facepoint(vx: float, vy: float, tolerance: float = 1e-6) -> int:
            """Find the nearest facepoint index within tolerance."""
            distances = np.sqrt((facepoints_coords[:, 0] - vx)**2 + (facepoints_coords[:, 1] - vy)**2)
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]

            if min_dist > tolerance:
                raise ValueError(f"Vertex ({vx}, {vy}) not found within tolerance {tolerance}. Nearest point is {min_dist} away.")

            return min_dist_idx

        # 3. For each face center, find its polygon and extract vertex indices
        face_nodes_list = []
        for x, y in face_center_coords:
            center_point = Point((x, y))
            # Get candidate polygons using spatial index
            possible_matches_idx = list(spatial_index.intersection(center_point.bounds))
            possible_matches = mesh_cells.iloc[possible_matches_idx]

            # Find which polygon actually contains the point
            match_found = False
            for poly_geom in possible_matches.geometry:
                if poly_geom.contains(center_point):
                    # Extract vertices from polygon exterior (excluding last duplicate point)
                    vertices = list(poly_geom.exterior.coords)[:-1]

                    # Map vertices to facepoints indices using nearest neighbor search
                    vertex_indices = []
                    for vx, vy in vertices:
                        idx = find_nearest_facepoint(vx, vy)
                        vertex_indices.append(idx)

                    face_nodes_list.append(vertex_indices)
                    match_found = True
                    break
            assert match_found, f"No containing polygon found for node at ({x}, {y})"

        # Convert to padded numpy array with NaN for different vertex counts
        max_vertices = max(len(face) for face in face_nodes_list)
        face_nodes = np.full((len(face_nodes_list), max_vertices), np.nan)
        for i, vertex_indices in enumerate(face_nodes_list):
            face_nodes[i, :len(vertex_indices)] = vertex_indices

        return face_nodes

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
        fp_geom = [Point(x, y) for x, y in zip(self.node_x, self.node_y)]
        fp_gdf = gpd.GeoDataFrame({
            'X': self.node_x,
            'Y': self.node_y,
            'FP_index': np.arange(len(self.node_x))
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
                    (self.node_x[from_idx], self.node_y[from_idx]),
                    (self.node_x[to_idx], self.node_y[to_idx])
                ])
                faces_list.append({
                    'face_index': i,
                    'from': from_idx,
                    'to': to_idx,
                    'geometry': edge_geom
                })
        faces_gdf = gpd.GeoDataFrame(faces_list)
        return faces_gdf

