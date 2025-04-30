import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import logging

# Configurar logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SfMReconstructor:
    """
    Clase para la reconstrucción 3D de una superficie a partir de imágenes
    utilizando Structure from Motion (SfM).
    """
    
    def __init__(self, image_dir, output_dir='output', feature_method='sift', 
                 min_matches=20, match_ratio=0.7):
        """
        Inicialización de la clase SfMReconstructor.
        
        Args:
            image_dir: Directorio que contiene las imágenes de entrada
            output_dir: Directorio para guardar los resultados
            feature_method: Método para detectar características ('sift', 'orb')
            min_matches: Número mínimo de coincidencias para considerar dos imágenes conectadas
            match_ratio: Umbral para el ratio de coincidencias (test de Lowe)
        """
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.feature_method = feature_method
        self.min_matches = min_matches
        self.match_ratio = match_ratio
        
        # Crear directorio de salida si no existe
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Cargar imágenes
        self.images = []
        self.image_names = []
        self._load_images()
        
        # Estructuras para almacenar datos de reconstrucción
        self.keypoints = []  # Keypoints para cada imagen
        self.descriptors = []  # Descriptores de los keypoints
        self.matches = {}  # Coincidencias entre pares de imágenes
        self.cameras = {}  # Matrices de cámara estimadas
        self.point_cloud = None  # Nube de puntos 3D
        self.mesh = None  # Malla 3D final
        
    def _load_images(self):
        """Carga las imágenes desde el directorio especificado."""
        logger.info(f"Cargando imágenes desde: {self.image_dir}")
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        for filename in sorted(os.listdir(self.image_dir)):
            if filename.lower().endswith(valid_extensions):
                img_path = os.path.join(self.image_dir, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    self.images.append(img)
                    self.image_names.append(filename)
        
        logger.info(f"Se cargaron {len(self.images)} imágenes")
        
    def detect_features(self):
        """Detecta características (keypoints y descriptores) en todas las imágenes."""
        logger.info(f"Detectando características usando método: {self.feature_method}")
        
        # Inicializar detector de características
        if self.feature_method.lower() == 'sift':
            detector = cv2.SIFT_create()
        elif self.feature_method.lower() == 'orb':
            detector = cv2.ORB_create(nfeatures=3000)
        else:
            raise ValueError(f"Método de detección no compatible: {self.feature_method}")
        
        # Detectar características en cada imagen
        for i, img in enumerate(self.images):
            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detectar keypoints y calcular descriptores
            kp, des = detector.detectAndCompute(gray, None)
            
            self.keypoints.append(kp)
            self.descriptors.append(des)
            
            logger.info(f"Imagen {i+1}/{len(self.images)}: {len(kp)} keypoints detectados")
            
            # Visualizar keypoints (opcional)
            img_with_kp = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite(os.path.join(self.output_dir, f"keypoints_{i}.jpg"), img_with_kp)
            
        return True
        
    def match_features(self):
        """Establece correspondencias entre pares de imágenes."""
        logger.info("Estableciendo correspondencias entre imágenes...")
        
        # Elegir el matcher adecuado
        if self.feature_method.lower() == 'sift':
            matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:  # para ORB
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        # Comparar todas las combinaciones de pares de imágenes
        num_images = len(self.images)
        for i in range(num_images):
            for j in range(i+1, num_images):
                # Obtener coincidencias
                matches = matcher.knnMatch(self.descriptors[i], self.descriptors[j], k=2)
                
                # Aplicar filtro de ratio (Lowe's ratio test)
                good_matches = []
                for m, n in matches:
                    if m.distance < self.match_ratio * n.distance:
                        good_matches.append(m)
                
                # Guardar las coincidencias si superan el umbral mínimo
                if len(good_matches) >= self.min_matches:
                    self.matches[(i, j)] = good_matches
                    logger.info(f"Imágenes {i}-{j}: {len(good_matches)} coincidencias")
                    
                    # Visualizar coincidencias (opcional)
                    img_matches = cv2.drawMatches(
                        self.images[i], self.keypoints[i],
                        self.images[j], self.keypoints[j],
                        good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )
                    cv2.imwrite(os.path.join(self.output_dir, f"matches_{i}_{j}.jpg"), img_matches)
        
        logger.info(f"Total de pares con coincidencias: {len(self.matches)}")
        return len(self.matches) > 0
    
    def _estimate_essential_matrix(self, pts1, pts2, K):
        """
        Estima la matriz esencial entre un par de imágenes.
        
        Args:
            pts1: Puntos correspondientes en la primera imagen
            pts2: Puntos correspondientes en la segunda imagen
            K: Matriz de calibración de la cámara
            
        Returns:
            E: Matriz esencial
            mask: Máscara que indica inliers
        """
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        return E, mask
    
    def _decompose_essential_matrix(self, E, K, pts1, pts2):
        """
        Descompone la matriz esencial en rotación y traslación.
        
        Args:
            E: Matriz esencial
            K: Matriz de calibración de la cámara
            pts1: Puntos correspondientes en la primera imagen
            pts2: Puntos correspondientes en la segunda imagen
            
        Returns:
            R: Matriz de rotación 3x3
            t: Vector de traslación
            triangulated_points: Puntos 3D triangulados
        """
        # Recuperar R, t de la matriz esencial
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
        
        # Convertir a matriz de proyección
        P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
        P2 = np.dot(K, np.hstack((R, t)))
        
        # Triangular puntos 3D
        pts1_homo = cv2.convertPointsToHomogeneous(pts1)[:,0,:]
        pts2_homo = cv2.convertPointsToHomogeneous(pts2)[:,0,:]
        
        def triangulate_point(p1, p2):
            A = np.zeros((4, 4))
            A[0] = p1[0] * P1[2] - P1[0]
            A[1] = p1[1] * P1[2] - P1[1]
            A[2] = p2[0] * P2[2] - P2[0]
            A[3] = p2[1] * P2[2] - P2[1]
            
            _, _, vt = np.linalg.svd(A)
            return vt[-1]
        
        triangulated_points = []
        for i in range(len(pts1)):
            point_3d = triangulate_point(pts1_homo[i], pts2_homo[i])
            point_3d = point_3d / point_3d[3]  # Normalizar coordenadas homogéneas
            triangulated_points.append(point_3d[:3])
        
        return R, t, np.array(triangulated_points)
    
    def reconstruct_initial_pair(self):
        """
        Reconstruye la escena inicial a partir del par de imágenes con más coincidencias.
        
        Returns:
            Verdadero si la reconstrucción inicial fue exitosa
        """
        logger.info("Iniciando reconstrucción del par inicial...")
        
        # Encontrar el par de imágenes con más coincidencias
        best_pair = max(self.matches.keys(), key=lambda k: len(self.matches[k]))
        idx1, idx2 = best_pair
        
        logger.info(f"Par inicial seleccionado: imágenes {idx1}-{idx2} con {len(self.matches[best_pair])} coincidencias")
        
        # Estimar matriz de calibración (aproximada si no se tiene una real)
        img_height, img_width = self.images[idx1].shape[:2]
        focal_length = 1.2 * max(img_height, img_width)  # Aproximación de distancia focal
        K = np.array([
            [focal_length, 0, img_width/2],
            [0, focal_length, img_height/2],
            [0, 0, 1]
        ])
        
        # Obtener puntos correspondientes
        matches = self.matches[best_pair]
        pts1 = np.float32([self.keypoints[idx1][m.queryIdx].pt for m in matches])
        pts2 = np.float32([self.keypoints[idx2][m.trainIdx].pt for m in matches])
        
        # Estimar matriz esencial
        E, mask = self._estimate_essential_matrix(pts1, pts2, K)
        
        # Filtrar puntos usando la máscara de inliers
        inliers_mask = mask.ravel() == 1
        pts1_inliers = pts1[inliers_mask]
        pts2_inliers = pts2[inliers_mask]
        
        # Descomponer matriz esencial en R, t y triangular puntos
        R, t, points_3d = self._decompose_essential_matrix(E, K, pts1_inliers, pts2_inliers)
        
        # Guardar matrices de cámara
        self.cameras[idx1] = {
            'K': K,
            'R': np.eye(3),
            't': np.zeros((3, 1)),
            'P': np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
        }
        
        self.cameras[idx2] = {
            'K': K,
            'R': R,
            't': t,
            'P': np.dot(K, np.hstack((R, t)))
        }
        
        # Inicializar nube de puntos
        self.point_cloud = {
            'points': points_3d,
            'colors': np.zeros((len(points_3d), 3)),  # Colores pendientes
            'visibility': {}  # Para cada punto, lista de cámaras donde es visible
        }
        
        # Registrar visibilidad de puntos
        for i, (is_inlier, match) in enumerate(zip(inliers_mask, matches)):
            if is_inlier:
                point_idx = sum(inliers_mask[:i])
                if point_idx not in self.point_cloud['visibility']:
                    self.point_cloud['visibility'][point_idx] = {}
                
                self.point_cloud['visibility'][point_idx][idx1] = match.queryIdx
                self.point_cloud['visibility'][point_idx][idx2] = match.trainIdx
        
        # Visualizar reconstrucción inicial
        self._visualize_point_cloud(points_3d, "initial_reconstruction.ply")
        
        logger.info(f"Reconstrucción inicial completada con {len(points_3d)} puntos 3D")
        return True
    
    def add_more_views(self):
        """
        Añade más vistas a la reconstrucción, registrando las cámaras una por una.
        
        Returns:
            Número de cámaras registradas exitosamente
        """
        logger.info("Añadiendo más vistas a la reconstrucción...")
        
        # Cámaras ya registradas
        registered_cameras = set(self.cameras.keys())
        remaining_cameras = set(range(len(self.images))) - registered_cameras
        
        while remaining_cameras:
            best_camera = None
            best_score = 0
            
            # Encontrar la mejor cámara para registrar
            for cam_idx in remaining_cameras:
                score = 0
                for reg_idx in registered_cameras:
                    if (cam_idx, reg_idx) in self.matches:
                        score += len(self.matches[(cam_idx, reg_idx)])
                    elif (reg_idx, cam_idx) in self.matches:
                        score += len(self.matches[(reg_idx, cam_idx)])
                
                if score > best_score:
                    best_score = score
                    best_camera = cam_idx
            
            if best_camera is None or best_score < self.min_matches:
                logger.info("No se encontraron más cámaras para registrar")
                break
            
            # Registrar la mejor cámara
            if self._register_camera(best_camera, registered_cameras):
                registered_cameras.add(best_camera)
                remaining_cameras.remove(best_camera)
                logger.info(f"Cámara {best_camera} registrada exitosamente")
            else:
                logger.warning(f"No se pudo registrar la cámara {best_camera}")
                remaining_cameras.remove(best_camera)
        
        logger.info(f"Total de cámaras registradas: {len(registered_cameras)}")
        
        # Bundle adjustment global
        if len(registered_cameras) > 2:
            self._global_bundle_adjustment()
        
        # Visualizar la reconstrucción final
        points = self.point_cloud['points']
        self._visualize_point_cloud(points, "final_reconstruction.ply")
        
        return len(registered_cameras)
    
    def _register_camera(self, new_camera_idx, registered_cameras):
        """
        Registra una nueva cámara a la reconstrucción existente.
        
        Args:
            new_camera_idx: Índice de la cámara a registrar
            registered_cameras: Conjunto de cámaras ya registradas
            
        Returns:
            True si el registro fue exitoso
        """
        # Recopilar correspondencias 2D-3D
        points_2d = []
        points_3d = []
        
        for point_idx, visibility in self.point_cloud['visibility'].items():
            for reg_cam in registered_cameras:
                if reg_cam in visibility:
                    # Buscar correspondencia en la nueva cámara
                    for pair in [(new_camera_idx, reg_cam), (reg_cam, new_camera_idx)]:
                        if pair in self.matches:
                            matches = self.matches[pair]
                            
                            if pair[0] == new_camera_idx:  # nueva_camara -> reg_cam
                                for match in matches:
                                    if match.trainIdx == visibility[reg_cam]:
                                        kp = self.keypoints[new_camera_idx][match.queryIdx].pt
                                        points_2d.append(kp)
                                        points_3d.append(self.point_cloud['points'][point_idx])
                                        break
                            else:  # reg_cam -> nueva_camara
                                for match in matches:
                                    if match.queryIdx == visibility[reg_cam]:
                                        kp = self.keypoints[new_camera_idx][match.trainIdx].pt
                                        points_2d.append(kp)
                                        points_3d.append(self.point_cloud['points'][point_idx])
                                        break
        
        if len(points_2d) < 6:
            logger.warning(f"Insuficientes correspondencias 2D-3D para cámara {new_camera_idx}: {len(points_2d)}")
            return False
        
        # Convertir a formato numpy
        points_2d = np.array(points_2d, dtype=np.float32)
        points_3d = np.array(points_3d, dtype=np.float32)
        
        # Obtener K de una cámara existente (asumimos calibración idéntica)
        K = self.cameras[list(registered_cameras)[0]]['K']
        
        # Resolver el problema PnP
        _, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d, K, None)
        
        if inliers is None or len(inliers) < 6:
            logger.warning(f"PnP falló para cámara {new_camera_idx}")
            return False
        
        # Convertir rvec a matriz de rotación
        R, _ = cv2.Rodrigues(rvec)
        
        # Guardar matriz de cámara
        self.cameras[new_camera_idx] = {
            'K': K,
            'R': R,
            't': tvec,
            'P': np.dot(K, np.hstack((R, tvec)))
        }
        
        # Triangular nuevos puntos
        self._triangulate_new_points(new_camera_idx, registered_cameras)
        
        return True
    
    def _triangulate_new_points(self, new_camera_idx, registered_cameras):
        """
        Triangula nuevos puntos con la nueva cámara registrada.
        
        Args:
            new_camera_idx: Índice de la cámara recién registrada
            registered_cameras: Conjunto de cámaras ya registradas
        """
        new_P = self.cameras[new_camera_idx]['P']
        
        for reg_cam in registered_cameras:
            pair = None
            if (new_camera_idx, reg_cam) in self.matches:
                pair = (new_camera_idx, reg_cam)
                idx1, idx2 = new_camera_idx, reg_cam
            elif (reg_cam, new_camera_idx) in self.matches:
                pair = (reg_cam, new_camera_idx)
                idx1, idx2 = reg_cam, new_camera_idx
            
            if pair is None:
                continue
                
            matches = self.matches[pair]
            reg_P = self.cameras[reg_cam]['P']
            
            # Recopilar puntos correspondientes
            pts1 = []
            pts2 = []
            match_indices = []
            
            for i, match in enumerate(matches):
                # Comprobar si este punto ya está en la reconstrucción
                is_reconstructed = False
                for point_idx, visibility in self.point_cloud['visibility'].items():
                    if idx1 in visibility and visibility[idx1] == match.queryIdx:
                        # Actualizar visibilidad para la nueva cámara
                        self.point_cloud['visibility'][point_idx][idx2] = match.trainIdx
                        is_reconstructed = True
                        break
                
                if not is_reconstructed:
                    kp1 = self.keypoints[idx1][match.queryIdx].pt
                    kp2 = self.keypoints[idx2][match.trainIdx].pt
                    pts1.append(kp1)
                    pts2.append(kp2)
                    match_indices.append(i)
            
            if not pts1:
                continue
                
            # Triangular nuevos puntos
            pts1 = np.array(pts1, dtype=np.float32)
            pts2 = np.array(pts2, dtype=np.float32)
            
            # Convertir a coordenadas homogéneas
            pts1_homo = cv2.convertPointsToHomogeneous(pts1)[:,0,:]
            pts2_homo = cv2.convertPointsToHomogeneous(pts2)[:,0,:]
            
            # Triangular puntos
            for i in range(len(pts1)):
                p1 = pts1_homo[i]
                p2 = pts2_homo[i]
                
                A = np.zeros((4, 4))
                A[0] = p1[0] * reg_P[2] - reg_P[0]
                A[1] = p1[1] * reg_P[2] - reg_P[1]
                A[2] = p2[0] * new_P[2] - new_P[0]
                A[3] = p2[1] * new_P[2] - new_P[1]
                
                _, _, vt = np.linalg.svd(A)
                point_3d = vt[-1]
                point_3d = point_3d / point_3d[3]  # Normalizar coordenadas homogéneas
                
                # Comprobar si el punto está delante de ambas cámaras
                cam1_center = -np.dot(self.cameras[idx1]['R'].T, self.cameras[idx1]['t'])
                cam2_center = -np.dot(self.cameras[idx2]['R'].T, self.cameras[idx2]['t'])
                
                v1 = point_3d[:3] - cam1_center.flatten()
                v2 = point_3d[:3] - cam2_center.flatten()
                
                # Comprobar direcciones utilizando productos punto con las direcciones de vista
                if np.dot(self.cameras[idx1]['R'][2], v1) > 0 and np.dot(self.cameras[idx2]['R'][2], v2) > 0:
                    # Añadir nuevo punto a la reconstrucción
                    new_point_idx = len(self.point_cloud['points'])
                    self.point_cloud['points'] = np.vstack([self.point_cloud['points'], point_3d[:3]])
                    self.point_cloud['colors'] = np.vstack([self.point_cloud['colors'], [0, 0, 0]])
                    
                    # Registrar visibilidad
                    self.point_cloud['visibility'][new_point_idx] = {
                        idx1: matches[match_indices[i]].queryIdx,
                        idx2: matches[match_indices[i]].trainIdx
                    }
    
    def _global_bundle_adjustment(self):
        """Realiza un ajuste global de haces para refinar la reconstrucción."""
        logger.info("Iniciando bundle adjustment global...")
        
        # Identificar puntos visibles en cada cámara
        observations = {}
        for point_idx, visibility in self.point_cloud['visibility'].items():
            point_3d = self.point_cloud['points'][point_idx]
            
            for cam_idx, keypoint_idx in visibility.items():
                kp = self.keypoints[cam_idx][keypoint_idx].pt
                
                if cam_idx not in observations:
                    observations[cam_idx] = {'points_3d': [], 'points_2d': []}
                
                observations[cam_idx]['points_3d'].append(point_3d)
                observations[cam_idx]['points_2d'].append(kp)
        
        # Refinar cada cámara
        for cam_idx, obs in observations.items():
            if len(obs['points_2d']) < 10:
                continue
                
            points_3d = np.array(obs['points_3d'], dtype=np.float32)
            points_2d = np.array(obs['points_2d'], dtype=np.float32)
            
            # Refinar pose usando PnP
            K = self.cameras[cam_idx]['K']
            rvec, _ = cv2.Rodrigues(self.cameras[cam_idx]['R'])
            tvec = self.cameras[cam_idx]['t']
            
            _, rvec, tvec = cv2.solvePnP(
                points_3d, points_2d, K, None,
                rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE
            )
            
            # Actualizar matrices de cámara
            R, _ = cv2.Rodrigues(rvec)
            self.cameras[cam_idx]['R'] = R
            self.cameras[cam_idx]['t'] = tvec
            self.cameras[cam_idx]['P'] = np.dot(K, np.hstack((R, tvec)))
        
        logger.info("Bundle adjustment completado")
    
    def _visualize_point_cloud(self, points, filename="point_cloud.ply"):
        """
        Visualiza y guarda una nube de puntos.
        
        Args:
            points: Array numpy con puntos 3D
            filename: Nombre del archivo para guardar la nube de puntos
        """
        # Crear nube de puntos Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Asignar colores (por defecto)
        colors = np.ones_like(points) * [0.5, 0.5, 0.5]  # Gris por defecto
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Guardar nube de puntos
        output_path = os.path.join(self.output_dir, filename)
        o3d.io.write_point_cloud(output_path, pcd)
        
        logger.info(f"Nube de puntos guardada en: {output_path}")
        
        return pcd
    
    def create_dense_point_cloud(self):
        """
        Genera una nube de puntos densa utilizando técnicas de MVS.
        
        Returns:
            Verdadero si la generación fue exitosa
        """
        logger.info("Generando nube de puntos densa...")
        
        # Para una implementación completa de MVS se necesitaría una biblioteca especializada
        # como COLMAP o MVE. Aquí presentaremos una versión simplificada.
        
        # Utilizar la nube de puntos dispersa como base
        if self.point_cloud is None or len(self.point_cloud['points']) == 0:
            logger.error("No hay reconstrucción inicial disponible")
            return False
        
        sparse_points = self.point_cloud['points']
        
        # Extraer datos de cámara para procesamiento MVS
        camera_data = []
        for cam_idx, cam in self.cameras.items():
            image = self.images[cam_idx]
            
            # Crear un diccionario con datos relevantes
            camera_data.append({
                'idx': cam_idx,
                'image': image,
                'K': cam['K'],
                'R': cam['R'],
                't': cam['t'],
                'P': cam['P'],
                'depth_map': None  # Se calculará después
            })
        
        # Densificación básica utilizando proyección y búsqueda de correspondencias
        # (Para una implementación real, esto debería ser un algoritmo MVS completo)
        dense_points = self._basic_densification(sparse_points, camera_data)
        
        # Guardar nube de puntos densa
        dense_pcd = self._visualize_point_cloud(dense_points, "dense_point_cloud.ply")
        
        # Actualizar la nube de puntos
        self.point_cloud['dense_points'] = dense_points
        
        logger.info(f"Nube de puntos densa creada con {len(dense_points)} puntos")
        return True
    
    def _basic_densification(self, sparse_points, camera_data, patch_size=7, step=3):
        """
        Implementación básica de densificación de la nube de puntos.
        
        Args:
            sparse_points: Nube de puntos dispersa
            camera_data: Datos de las cámaras registradas
            patch_size: Tamaño del parche para búsqueda de correspondencias
            step: Paso para submuestreo de píxeles
            
        Returns:
            Array numpy con puntos 3D densos
        """
        dense_points = []
        
        # Usar los primeros dos pares de cámaras como base
        if len(camera_data) < 2:
            return sparse_points
            
        cam1 = camera_data[0]
        cam2 = camera_data[1]
        
        # Obtener imágenes en escala de grises
        gray1 = cv2.cvtColor(cam1['image'], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cam2['image'], cv2.COLOR_BGR2GRAY)
        
        h, w = gray1.shape
        
        # Para cada píx