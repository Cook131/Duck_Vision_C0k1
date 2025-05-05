import cv2
import numpy as np

# Parámetros
video_path = '/Users/jorgenajera/Documents/Duck_Vision_/Canny2/badToTheBone.mp4'
target_fps = 12
canny_thresh1 = 100
canny_thresh2 = 200
frame_delay = 50  # Delay en milisegundos entre frames

# Inicializa el filtro de Kalman de OpenCV
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(round(video_fps / target_fps))

frame_count = 0
kalman_initialized = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Salta frames para alcanzar el target_fps
    if frame_count % frame_interval != 0:
        frame_count += 1
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2)

    # Crear una visualización en escala de grises
    # Convertir el frame a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Mezclar el frame en gris con los bordes
    display = cv2.addWeighted(gray_frame, 0.3, edges, 0.7, 0)

    # Encuentra contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Calcula el centroide del contorno
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        # Inicializa el filtro de Kalman con la primera medición
        if not kalman_initialized:
            kalman.statePre = np.array([[cx],[cy],[0],[0]], np.float32)
            kalman.statePost = np.array([[cx],[cy],[0],[0]], np.float32)
            kalman_initialized = True

        # Predicción y corrección del filtro de Kalman
        prediction = kalman.predict()
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        estimated = kalman.correct(measurement)

        # Dibuja el contorno y el centroide
        cv2.drawContours(display, [cnt], -1, (255,255,255), 1)  # Contorno en blanco
        cv2.circle(display, (cx, cy), 3, (255,255,255), -1)      # Centroide en blanco

    cv2.imshow('Canny Edges', display)
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):  # Aumentado el delay entre frames
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
