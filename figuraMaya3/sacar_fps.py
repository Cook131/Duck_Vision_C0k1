import cv2
import os

# Cambia estos nombres si tu carpeta/video tienen otro nombre
carpeta = "/Users/jorgenajera/Documents/Duck_vision_/imagenes_3D"
video_nombre = "video_escultura.mp4"  # Cambia por el nombre real de tu video

# Ruta completa al video
video_path = os.path.join(carpeta, video_nombre)

# Crea la carpeta si no existe
if not os.path.exists(carpeta):
    os.makedirs(carpeta)

# Abre el video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("No se pudo abrir el video:", video_path)
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS detectados: {fps}")

# Queremos 30 fps
fps_deseados = 30
frame_interval = int(round(fps / fps_deseados)) if fps > fps_deseados else 1

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % frame_interval == 0:
        nombre_frame = f"frame_{saved_count:04d}.jpg"
        cv2.imwrite(os.path.join(carpeta, nombre_frame), frame)
        saved_count += 1
    frame_count += 1

cap.release()
print(f"Frames guardados: {saved_count}")
