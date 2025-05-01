from deteccion_limites import SfMReconstructor  # importa la clase del archivo donde la guardaste

def main():
    # Ruta a la carpeta con las imágenes
    image_dir = '/Users/jorgenajera/Documents/Duck_Vision_/imagenes_bici'
    
    # Crear objeto reconstructor
    reconstructor = SfMReconstructor(image_dir, output_dir='/Users/jorgenajera/Documents/Duck_Vision_/output', feature_method='sift')
    
    # Detectar características en todas las imágenes
    reconstructor.detect_features()
    
    # Emparejar características entre imágenes
    if not reconstructor.match_features():
        print("No se encontraron suficientes coincidencias entre imágenes.")
        return
    
    # Reconstruir la escena inicial con el mejor par
    if not reconstructor.reconstruct_initial_pair():
        print("Error en la reconstrucción inicial.")
        return
    
    # Añadir más vistas para completar la reconstrucción
    num_cameras = reconstructor.add_more_views()
    print(f"Cámaras registradas: {num_cameras}")

if __name__ == '__main__':
    main()
