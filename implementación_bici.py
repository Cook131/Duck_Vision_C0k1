from deteccion_limites import SfMReconstructor  # importa la clase del archivo donde la guardaste

def main():
    # Ruta a la carpeta con las imágenes
    image_dir = '/Users/jorgenajera/Documents/Duck_Vision_/imagenes_bici'
    
    # Crear objeto reconstructor
    reconstructor = SfMReconstructor(image_dir, output_dir='/Users/jorgenajera/Documents/Duck_Vision_/output', feature_method='sift')
    
    # Detectar características en todas las imágenes
    reconstructor.detect_features()
    
    # NUEVO: Probar diferentes parámetros RANSAC
    best_params = reconstructor.test_ransac_parameters(
        thresholds=[0.5, 0.7, 1.0, 1.5, 2.0],
        probabilities=[0.95, 0.99],
        sample_pairs=3  # Prueba con los 3 mejores pares
    )
    
    # Usar los mejores parámetros encontrados para el emparejamiento final
    if not reconstructor.match_features(
        ransac_threshold=best_params['threshold'],
        ransac_probability=best_params['probability']
    ):
        print("No se encontraron suficientes coincidencias entre imágenes.")
        return
    
    # Continuar con la reconstrucción usando los matches mejorados
    if not reconstructor.reconstruct_initial_pair():
        print("Error en la reconstrucción inicial.")
        return
    
    # Añadir más vistas
    reconstructor.add_more_views()
    
    print("Reconstrucción 3D completada.")

if __name__ == '__main__':
    main()
