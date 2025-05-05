import open3d as o3d

pcd = o3d.io.read_point_cloud("/Users/jorgenajera/Documents/Duck_Vision_/figuraMaya3/output/final_reconstruction.ply")
pcd_initial = o3d.io.read_point_cloud("/Users/jorgenajera/Documents/Duck_Vision_/figuraMaya3/output/initial_reconstruction.ply")
o3d.visualization.draw([pcd, pcd_initial])  # En Jupyter, usa draw() en vez de draw_geometries()
