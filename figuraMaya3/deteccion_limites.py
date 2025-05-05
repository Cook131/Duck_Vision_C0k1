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
        
    def match_features(self, ransac_threshold=0.7, ransac_probability=0.99):
        """
        Matches features between pairs of images using the specified feature detection method
        and filters matches using the RANSAC algorithm to find inliers.

        Parameters:
        -----------
        ransac_threshold : float, optional
            The maximum allowed reprojection error to treat a point pair as an inlier 
            during the RANSAC process. Default is 1.0.
        ransac_probability : float, optional
            The confidence level for the RANSAC algorithm to estimate the fundamental matrix. 
            Default is 0.99.

        Returns:
        --------
        bool
            True if at least one pair of images has sufficient inlier matches, False otherwise.

        Notes:
        ------
        - The method uses either SIFT or another feature detection method based on the 
          `self.feature_method` attribute to compute matches.
        - Matches are filtered using the fundamental matrix estimated with RANSAC.
        - Inlier matches are saved for each pair of images that meet the minimum match threshold.
        - Visualizations of the matches are saved as images in the specified output directory.
        - Logs the number of inlier matches for each image pair and the total number of pairs 
          with matches.
        """
        logger.info("Estableciendo correspondencias entre imágenes...")
        if self.feature_method.lower() == 'sift':
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        num_images = len(self.images)
        for i in range(num_images):
            for j in range(i+1, num_images):
                matches = matcher.match(self.descriptors[i], self.descriptors[j])
                matches = sorted(matches, key=lambda x: x.distance)
                if len(matches) >= self.min_matches:
                    pts1 = np.float32([self.keypoints[i][m.queryIdx].pt for m in matches])
                    pts2 = np.float32([self.keypoints[j][m.trainIdx].pt for m in matches])
                    F, mask = cv2.findFundamentalMat(
                        pts1, pts2, cv2.FM_RANSAC,
                        ransacReprojThreshold=ransac_threshold,
                        confidence=ransac_probability
                    )
                    if F is not None and mask is not None:
                        mask = mask.ravel().astype(bool)
                        inlier_matches = [matches[k] for k in range(len(matches)) if mask[k]]
                        if len(inlier_matches) >= self.min_matches:
                            self.matches[(i, j)] = inlier_matches
                            logger.info(f"Imágenes {i}-{j}: {len(inlier_matches)} inliers de {len(matches)} matches iniciales")
                            img_matches = cv2.drawMatches(
                                self.images[i], self.keypoints[i],
                                self.images[j], self.keypoints[j],
                                inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                            )
                            cv2.imwrite(os.path.join(self.output_dir, f"matches_{i}_{j}.jpg"), img_matches)
        logger.info(f"Total de pares con matches: {len(self.matches)}")
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
        focal_length = max(img_height, img_width)
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

    def test_ransac_parameters(self, thresholds=[0.5, 1.0, 1.5, 2.0], 
                          probabilities=[0.90, 0.95, 0.99], 
                          sample_pairs=3):
        """
        Prueba diferentes combinaciones de umbrales RANSAC y probabilidades
        para encontrar la configuración óptima para el emparejamiento de características.
        
        Args:
            thresholds: Lista de umbrales de reproyección a probar (en píxeles)
            probabilities: Lista de probabilidades RANSAC a probar
            sample_pairs: Número de pares de imágenes a probar (los con más matches iniciales)
            
        Returns:
            best_params: Diccionario con los mejores parámetros encontrados
        """
        logger.info("Iniciando prueba de parámetros RANSAC...")
        
        # Detectar características si no se ha hecho ya
        if not self.keypoints or not self.descriptors:
            self.detect_features()
        
        # Preparar matcher
        if self.feature_method.lower() == 'sift':
            matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        # Encontrar los pares de imágenes con más coincidencias potenciales
        num_images = len(self.images)
        pairs_to_test = []
        for i in range(num_images):
            for j in range(i+1, num_images):
                # Obtener matches iniciales con ratio test
                matches = matcher.knnMatch(self.descriptors[i], self.descriptors[j], k=2)
                good_matches = []
                for m, n in matches:
                    if m.distance < self.match_ratio * n.distance:
                        good_matches.append(m)
                
                if len(good_matches) >= self.min_matches:
                    pairs_to_test.append((i, j, good_matches, len(good_matches)))
        
        # Ordenar por número de matches y seleccionar los mejores
        pairs_to_test.sort(key=lambda x: x[3], reverse=True)
        pairs_to_test = pairs_to_test[:sample_pairs]
        
        # Estimar matriz de calibración (aproximada)
        img_height, img_width = self.images[0].shape[:2]
        focal_length = max(img_height, img_width)
        K = np.array([
            [focal_length, 0, img_width/2],
            [0, focal_length, img_height/2],
            [0, 0, 1]
        ])
        
        # Preparar para guardar resultados
        results = {}
        
        # Directorio para guardar visualizaciones
        test_dir = os.path.join(self.output_dir, "ransac_tests")
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        
        # Probar cada combinación de parámetros
        for threshold in thresholds:
            for probability in probabilities:
                config_name = f"thresh_{threshold}_prob_{probability}"
                results[config_name] = {
                    'threshold': threshold,
                    'probability': probability,
                    'inlier_ratios': [],
                    'total_inliers': 0,
                    'pair_results': []
                }
                
                logger.info(f"Probando configuración: umbral={threshold}, probabilidad={probability}")
                
                # Probar en cada par de imágenes seleccionado
                for idx, (i, j, good_matches, _) in enumerate(pairs_to_test):
                    # Extraer puntos correspondientes
                    pts1 = np.float32([self.keypoints[i][m.queryIdx].pt for m in good_matches])
                    pts2 = np.float32([self.keypoints[j][m.trainIdx].pt for m in good_matches])
                    
                    # Aplicar filtrado geométrico con RANSAC
                    F, mask = cv2.findFundamentalMat(
                        pts1, pts2, 
                        method=cv2.FM_RANSAC, 
                        ransacReprojThreshold=threshold, 
                        confidence=probability
                    )
                    
                    # Si no se pudo calcular F o no hubo inliers
                    if F is None or mask is None:
                        inlier_ratio = 0
                        inlier_count = 0
                        filtered_matches = []
                    else:
                        mask = mask.ravel().astype(bool)
                        inlier_count = np.sum(mask)
                        inlier_ratio = inlier_count / len(good_matches) if len(good_matches) > 0 else 0
                        filtered_matches = [m for idx, m in enumerate(good_matches) if mask[idx]]
                    
                    # Guardar resultados de este par
                    results[config_name]['inlier_ratios'].append(inlier_ratio)
                    results[config_name]['total_inliers'] += inlier_count
                    results[config_name]['pair_results'].append({
                        'pair': (i, j),
                        'inlier_ratio': inlier_ratio,
                        'inlier_count': inlier_count,
                        'total_matches': len(good_matches)
                    })
                    
                    # Visualizar y guardar imagen de matches filtrados
                    img_matches = cv2.drawMatches(
                        self.images[i], self.keypoints[i],
                        self.images[j], self.keypoints[j],
                        filtered_matches, None, 
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )
                    
                    vis_filename = f"{config_name}_pair_{i}_{j}.jpg"
                    cv2.imwrite(os.path.join(test_dir, vis_filename), img_matches)
                
                # Calcular métricas globales para esta configuración
                results[config_name]['avg_inlier_ratio'] = np.mean(results[config_name]['inlier_ratios'])
        
        # Determinar la mejor configuración
        # Criterio: mayor ratio de inliers manteniendo un número razonable de matches
        best_config = max(results.keys(), key=lambda k: results[k]['avg_inlier_ratio'])
        best_threshold = results[best_config]['threshold']
        best_probability = results[best_config]['probability']
        
        logger.info(f"Mejor configuración encontrada: umbral={best_threshold}, "
                    f"probabilidad={best_probability}, "
                    f"ratio inliers promedio={results[best_config]['avg_inlier_ratio']:.4f}")
        
        # Crear informe visual comparativo
        self._create_parameter_test_report(results, test_dir)
        
        return {
            'threshold': best_threshold,
            'probability': best_probability,
            'results': results
        }

    def _create_parameter_test_report(self, results, test_dir):
        """Crea un informe visual comparativo de las diferentes configuraciones probadas."""
        # Preparar datos para gráficos
        configs = list(results.keys())
        avg_ratios = [results[c]['avg_inlier_ratio'] for c in configs]
        total_inliers = [results[c]['total_inliers'] for c in configs]
        
        # Crear figura para el informe
        plt.figure(figsize=(12, 10))
        
        # Gráfico de ratio promedio de inliers
        plt.subplot(2, 1, 1)
        bars = plt.bar(configs, avg_ratios)
        plt.title('Ratio promedio de inliers por configuración')
        plt.xticks(rotation=45)
        plt.ylabel('Ratio promedio de inliers')
        
        # Añadir valores como etiquetas
        for bar, ratio in zip(bars, avg_ratios):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{ratio:.2f}', ha='center', va='bottom')
        
        # Gráfico de total de inliers
        plt.subplot(2, 1, 2)
        bars = plt.bar(configs, total_inliers)
        plt.title('Total de inliers por configuración')
        plt.xticks(rotation=45)
        plt.ylabel('Número total de inliers')
        
        # Añadir valores como etiquetas
        for bar, count in zip(bars, total_inliers):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(test_dir, 'parameter_comparison.png'))
        plt.close()
        
        logger.info(f"Informe de comparación guardado en {os.path.join(test_dir, 'parameter_comparison.png')}")


# Ejemplo de uso:
# reconstructor = SfMReconstructor('ruta/a/imagenes')
# reconstructor.detect_features()
# reconstructor.match_features()
# reconstructor.reconstruct_initial_pair()
# reconstructor.add_more_views()

size_mb = os.path.getsize("/Users/jorgenajera/Documents/Duck_Vision_/figuraMaya3/output/final_reconstruction.ply") / (1024 * 1024)
print(f"Tamaño del archivo PLY: {size_mb:.2f} MB")

def visualizar_dos_nubes(ply1, ply2):
    if not os.path.exists(ply1) or not os.path.exists(ply2):
        print("No se encuentran ambos archivos PLY.")
        return
    pcd1 = o3d.io.read_point_cloud(ply1)
    pcd2 = o3d.io.read_point_cloud(ply2)
    print(f"Nube inicial: {len(pcd1.points)} puntos")
    print(f"Nube final: {len(pcd2.points)} puntos")
    o3d.visualization.draw_geometries([pcd1.paint_uniform_color([1,0,0]), pcd2.paint_uniform_color([0,1,0])])

if __name__ == '__main__':
    ply1 = "/Users/jorgenajera/Documents/Duck_Vision_/figuraMaya3/output/initial_reconstruction.ply"
    ply2 = "/Users/jorgenajera/Documents/Duck_Vision_/figuraMaya3/output/final_reconstruction.ply"
    visualizar_dos_nubes(ply1, ply2)