from deteccion_limites import SfMReconstructor  # importa la clase del archivo donde la guardaste

def main():
    """
    Main function to perform 3D reconstruction using Structure from Motion (SfM).
    This function initializes the SfMReconstructor with the specified image directory,
    output directory, and feature detection method. It performs the following steps:
    1. Detects features in all images within the specified directory.
    2. Matches features between images using RANSAC with a specified threshold and probability.
    3. Reconstructs the initial pair of images to establish the 3D structure.
    4. Adds more views to the reconstruction to complete the 3D model.
    If any step fails (e.g., insufficient matches or reconstruction errors), the process
    terminates with an appropriate error message.
    Note:
        - Ensure that the image directory contains valid images for processing.
        - The output directory will store the results of the reconstruction.
    Returns:
        None
    """
    # Ruta a la carpeta con las imágenes
    image_dir = '/Users/jorgenajera/Documents/Duck_Vision_/figuraMaya3/imagenes_3D'
    
    # Crear objeto reconstructor
    reconstructor = SfMReconstructor(image_dir, output_dir='/Users/jorgenajera/Documents/Duck_Vision_/figuraMaya3/output', feature_method='sift')
    
    # Detectar características en todas las imágenes
    reconstructor = SfMReconstructor(image_dir, output_dir='output', feature_method='sift')
    reconstructor.detect_features()
    # Usa SOLO UN VALOR para el umbral y la probabilidad de RANSAC:
    if not reconstructor.match_features(ransac_threshold=1.0, ransac_probability=0.99):
        print("No se encontraron suficientes coincidencias entre imágenes.")
        return
    if not reconstructor.reconstruct_initial_pair():
        print("Error en la reconstrucción inicial.")
        return
    reconstructor.add_more_views()
    print("Reconstrucción 3D completada.")

if __name__ == '__main__':
    main()
