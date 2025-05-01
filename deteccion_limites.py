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
        
        # Matrices de proyección
        P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
        P2 = np.dot(K, np.hstack((R, t)))
        
        # Triangular puntos 3D con cv2.triangulatePoints (requiere puntos en formato homogéneo)
        pts1_h = cv2.undistortPoints(np.expand_dims(pts1, axis=1), K, None)
        pts2_h = cv2.undistortPoints(np.expand_dims(pts2, axis=1), K, None)
        
        pts4d_hom = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
        pts3d = (pts4d_hom[:3] / pts4d_hom[3]).T  # Convertir a coordenadas 3D
        
        return R, t, pts3d
    
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
            'visibility': {}  # Para cada punto, diccionario de cámaras donde es visible y el índice del keypoint
        }
        
        # Registrar visibilidad de puntos
        count = 0
        for i, (is_inlier, match) in enumerate(zip(inliers_mask, matches)):
            if is_inlier:
                self.point_cloud['visibility'][count] = {
                    idx1: match.queryIdx,
                    idx2: match.trainIdx
                }
                count += 1
        
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
            # Solo si el punto es visible en alguna cámara registrada
            visible_in_registered = any(cam in visibility for cam in registered_cameras)
            if not visible_in_registered:
                continue
            
            # Buscar correspondencia en la nueva cámara a través de matches
            found_2d = False
            for reg_cam in registered_cameras:
                if reg_cam in visibility:
                    # Buscar matches entre new_camera_idx y reg_cam
                    pairs = [(new_camera_idx, reg_cam), (reg_cam, new_camera_idx)]
                    for pair in pairs:
                        if pair in self.matches:
                            matches = self.matches[pair]
                            if pair[0] == new_camera_idx:
                                # new_camera_idx -> reg_cam
                                for match in matches:
                                    if match.trainIdx == visibility[reg_cam]:
                                        kp = self.keypoints[new_camera_idx][match.queryIdx].pt
                                        points_2d.append(kp)
                                        points_3d.append(self.point_cloud['points'][point_idx])
                                        found_2d = True
                                        break
                            else:
                                # reg_cam -> new_camera_idx
                                for match in matches:
                                    if match.queryIdx == visibility[reg_cam]:
                                        kp = self.keypoints[new_camera_idx][match.trainIdx].pt
                                        points_2d.append(kp)
                                        points_3d.append(self.point_cloud['points'][point_idx])
                                        found_2d = True
                                        break
                        if found_2d:
                            break
                if found_2d:
                    break
        
        if len(points_2d) < self.min_matches:
            logger.warning(f"No hay suficientes correspondencias 2D-3D para registrar la cámara {new_camera_idx}")
            return False
        
        points_2d = np.array(points_2d, dtype=np.float32)
        points_3d = np.array(points_3d, dtype=np.float32)
        
        # Obtener matriz de calibración de una cámara ya registrada (asumimos igual para todas)
        K = next(iter(self.cameras.values()))['K']
        
        # Estimar pose con solvePnPRansac
        success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d, K, None)
        if not success or inliers is None or len(inliers) < self.min_matches:
            logger.warning(f"solvePnPRansac falló para la cámara {new_camera_idx}")
            return False
        
        R, _ = cv2.Rodrigues(rvec)
        t = tvec
        
        # Guardar cámara
        self.cameras[new_camera_idx] = {
            'K': K,
            'R': R,
            't': t,
            'P': np.dot(K, np.hstack((R, t)))
        }
        
        # Actualizar visibilidad de puntos con la nueva cámara
        for idx in inliers.flatten():
            point_idx = idx
            if point_idx not in self.point_cloud['visibility']:
                self.point_cloud['visibility'][point_idx] = {}
            # Encontrar keypoint index en la nueva cámara
            # Buscamos el keypoint que corresponde a la 2D point in points_2d[idx]
            # Ya que points_2d se construyó con kp de new_camera_idx, guardamos ese índice
            # Pero no tenemos el índice directo, así que lo omitimos (podríamos mejorar)
            # Para simplicidad, guardamos el índice del keypoint más cercano
            # Aquí asumimos que points_2d[idx] corresponde a algún keypoint, lo guardamos como None
            self.point_cloud['visibility'][point_idx][new_camera_idx] = None
        
        logger.info(f"Cámara {new_camera_idx} registrada con éxito")
        return True
    
    def _global_bundle_adjustment(self):
        """
        Realiza un ajuste global (bundle adjustment) para optimizar poses y puntos 3D.
        Aquí se implementa una versión simplificada que solo optimiza puntos 3D,
        asumiendo poses fijas.
        """
        logger.info("Iniciando bundle adjustment global (simplificado)...")
        
        # Extraer datos para optimización
        points_3d = self.point_cloud['points']
        cameras = self.cameras
        visibility = self.point_cloud['visibility']
        
        # Construir vectores para optimización
        x0 = points_3d.flatten()
        
        def reprojection_residuals(x):
            pts = x.reshape((-1, 3))
            residuals = []
            for i, point in enumerate(pts):
                if i not in visibility:
                    continue
                for cam_idx, kp_idx in visibility[i].items():
                    cam = cameras[cam_idx]
                    K, R, t = cam['K'], cam['R'], cam['t']
                    P = np.dot(K, np.hstack((R, t)))
                    pt_h = np.hstack((point, 1))
                    proj = P @ pt_h
                    proj /= proj[2]
                    # Obtener punto 2D observado
                    if kp_idx is None:
                        continue  # Sin índice de keypoint, no podemos calcular residual
                    kp = self.keypoints[cam_idx][kp_idx].pt
                    residuals.extend(proj[:2] - kp)
            return residuals
        
        # Optimizar
        res = least_squares(reprojection_residuals, x0, verbose=2, ftol=1e-4, xtol=1e-4)
        
        # Actualizar puntos 3D
        self.point_cloud['points'] = res.x.reshape((-1, 3))
        logger.info("Bundle adjustment completado")
    
    def _visualize_point_cloud(self, points, filename):
        """
        Visualiza y guarda la nube de puntos 3D en formato PLY.
        
        Args:
            points: Array Nx3 de puntos 3D
            filename: Nombre del archivo PLY de salida
        """
        logger.info(f"Guardando nube de puntos en {filename}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(os.path.join(self.output_dir, filename), pcd)
        
        # Visualizar con Open3D (opcional)
        # o3d.visualization.draw_geometries([pcd])

# Ejemplo de uso:
# reconstructor = SfMReconstructor('ruta/a/imagenes')
# reconstructor.detect_features()
# reconstructor.match_features()
# reconstructor.reconstruct_initial_pair()
# reconstructor.add_more_views()