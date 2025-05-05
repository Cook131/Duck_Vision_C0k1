from deteccion_limites import SfMReconstructor
import os
import logging

def main():
    """
    Función principal para realizar la reconstrucción 3D usando Structure from Motion (SfM).
    Incluye manejo de errores y optimización de parámetros.
    """
    # Configurar logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Rutas
    image_dir = '/Users/jorgenajera/Documents/Duck_Vision_/figuraMaya3/imagenes_3D'
    output_dir = '/Users/jorgenajera/Documents/Duck_Vision_/figuraMaya3/output'
    
    # Asegurar que existe el directorio de salida
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Crear reconstructor con parámetros optimizados
        reconstructor = SfMReconstructor(
            image_dir=image_dir,
            output_dir=output_dir,
            feature_method='sift',
            min_matches=30,  # Aumentado para mejor calidad
            match_ratio=0.75  # Ajustado para mejor balance
        )
        
        # Paso 1: Detectar características
        logger.info("Detectando características...")
        reconstructor.detect_features()
        
        # Paso 2: Probar parámetros RANSAC
        logger.info("Probando parámetros RANSAC...")
        best_params = reconstructor.test_ransac_parameters(
            thresholds=[0.5, 1.0, 1.5],
            probabilities=[0.95, 0.99],
            sample_pairs=3
        )
        
        # Paso 3: Emparejar características con los mejores parámetros
        logger.info("Emparejando características...")
        if not reconstructor.match_features(
            ransac_threshold=best_params['threshold'],
            ransac_probability=best_params['probability']
        ):
            logger.error("No se encontraron suficientes coincidencias entre imágenes.")
            return
        
        # Paso 4: Reconstrucción inicial
        logger.info("Iniciando reconstrucción del par inicial...")
        if not reconstructor.reconstruct_initial_pair():
            logger.error("Error en la reconstrucción inicial.")
            return
        
        # Paso 5: Añadir más vistas
        logger.info("Añadiendo más vistas...")
        num_cameras = reconstructor.add_more_views()
        logger.info(f"Reconstrucción completada con {num_cameras} cámaras registradas.")
        
        # Paso 6: Verificar resultados
        final_ply = os.path.join(output_dir, "final_reconstruction.ply")
        if os.path.exists(final_ply):
            logger.info(f"Archivo PLY generado: {final_ply}")
            # Verificar tamaño del archivo
            size_mb = os.path.getsize(final_ply) / (1024 * 1024)
            logger.info(f"Tamaño del archivo PLY: {size_mb:.2f} MB")
        else:
            logger.error("No se generó el archivo PLY final.")
        
    except Exception as e:
        logger.error(f"Error durante la reconstrucción: {str(e)}")
        raise

if __name__ == '__main__':
    main()
