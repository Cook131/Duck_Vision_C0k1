import open3d as o3d
import numpy as np
import os

def visualize_point_cloud(ply_path, voxel_size=0.01, nb_neighbors=20, std_ratio=2.0):
    """
    Visualiza y procesa una nube de puntos PLY.
    
    Args:
        ply_path: Ruta al archivo PLY
        voxel_size: Tamaño del voxel para downsampling
        nb_neighbors: Número de vecinos para filtrado de outliers
        std_ratio: Ratio de desviación estándar para filtrado
    """
    if not os.path.exists(ply_path):
        print(f"Error: No se encontró el archivo {ply_path}")
        return
    
    # Cargar nube de puntos
    print(f"Cargando nube de puntos desde {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    
    if len(pcd.points) == 0:
        print("Error: La nube de puntos está vacía")
        return
    
    print(f"Puntos originales: {len(pcd.points)}")
    
    # Procesar nube de puntos
    # 1. Downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f"Puntos después de downsampling: {len(pcd_down.points)}")
    
    # 2. Eliminar outliers
    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                std_ratio=std_ratio)
    pcd_clean = pcd_down.select_by_index(ind)
    print(f"Puntos después de limpieza: {len(pcd_clean.points)}")
    
    # 3. Calcular normales
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # 4. Orientar normales consistentemente
    pcd_clean.orient_normals_consistent_tangent_plane(100)
    
    # Visualizar
    print("Visualizando nube de puntos...")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Añadir geometría
    vis.add_geometry(pcd_clean)
    
    # Configurar vista
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])  # Fondo negro
    opt.point_size = 2.0
    
    # Configurar controles
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    # Ejecutar visualización
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    # Ruta al archivo PLY
    ply_path = "/Users/jorgenajera/Documents/Duck_Vision_/figuraMaya3/output/final_reconstruction.ply"
    
    # Visualizar nube de puntos
    visualize_point_cloud(ply_path)
