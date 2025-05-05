import cv2
import os
import numpy as np

def extract_frames(video_path, output_dir, target_fps=30, min_angle_diff=5):
    """
    Extrae frames del video con espaciado angular consistente.
    
    Args:
        video_path: Ruta al video
        output_dir: Directorio de salida
        target_fps: FPS objetivo
        min_angle_diff: Diferencia angular mínima entre frames (grados)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video: {video_path}")
        return
    
    # Obtener propiedades del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video info: {fps} FPS, {total_frames} frames, {duration:.2f} segundos")
    
    # Calcular intervalo de frames para el FPS objetivo
    frame_interval = int(round(fps / target_fps)) if fps > target_fps else 1
    
    # Variables para tracking
    frame_count = 0
    saved_count = 0
    last_angle = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Aquí podrías implementar detección de ángulo si tienes un marcador
            # Por ahora, asumimos que el video gira uniformemente
            current_angle = (frame_count / total_frames) * 360
            
            if last_angle is None or abs(current_angle - last_angle) >= min_angle_diff:
                # Mejorar calidad de imagen
                frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
                
                # Guardar frame
                nombre_frame = f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(os.path.join(output_dir, nombre_frame), frame, 
                          [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                saved_count += 1
                last_angle = current_angle
                print(f"Frame {saved_count} guardado (ángulo: {current_angle:.1f}°)")
        
        frame_count += 1
    
    cap.release()
    print(f"\nProceso completado:")
    print(f"- Frames guardados: {saved_count}")
    print(f"- FPS efectivo: {saved_count/duration:.1f}")

if __name__ == '__main__':
    # Configuración
    video_path = "/Users/jorgenajera/Documents/Duck_vision_/imagenes_3D/video_escultura.mp4"
    output_dir = "/Users/jorgenajera/Documents/Duck_Vision_/figuraMaya3/imagenes_3D"
    
    # Extraer frames
    extract_frames(video_path, output_dir, target_fps=30, min_angle_diff=5)
