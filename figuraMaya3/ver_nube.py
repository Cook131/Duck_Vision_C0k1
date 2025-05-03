import open3d as o3d

pcd = o3d.io.read_point_cloud("output/final_reconstruction.ply")
o3d.visualization.draw([pcd])  # En Jupyter, usa draw() en vez de draw_geometries()
