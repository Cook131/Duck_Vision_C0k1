# import cv2

# # Leer la imagen original
# img = cv2.imread('dog.jpg')

# # Verificar si la imagen se cargó correctamente
# if img is None:
#     print("Error: no se pudo cargar la imagen.")
#     exit()

# # Convertir a escala de grises
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Aplicar desenfoque para reducir el ruido
# img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

# # Detección de bordes con Canny
# edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=200)

# # Mostrar el resultado
# cv2.imshow('Canny Edge Detection', edges)
# cv2.waitKey(0)

# # Guardar la imagen de bordes
# cv2.imwrite('edges_output.png', edges)

# # Cerrar todas las ventanas
# cv2.destroyAllWindows()


# import cv2

# # Cargar el video
# cap = cv2.VideoCapture('patitos.mp4')  # Cambia 'video.mp4' por el nombre de tu archivo de video

# # Verificar que el video se abra correctamente
# if not cap.isOpened():
#     print("Error: no se pudo abrir el video.")
#     exit()

# # Loop para leer cada frame
# while True:
#     ret, frame = cap.read()

#     if not ret:
#         print("Fin del video o error al leer frame.")
#         break

#     # Convertir frame a escala de grises
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Aplicar desenfoque
#     frame_blur = cv2.GaussianBlur(frame_gray, (3, 3), 0)

#     # Aplicar Canny
#     edges = cv2.Canny(frame_blur, 0, 200)

#     # Mostrar el frame procesado
#     cv2.imshow('Canny Edge Detection - Video', edges)

#     # Esperar 1ms y salir si se presiona 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Liberar recursos
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# cap = cv2.VideoCapture('patitos.mp4')

# if not cap.isOpened():
#     print("Error al abrir el video.")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blur, 50, 150)

#     # Detectar líneas
#     lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

#     if lines is not None:
#         for line in lines:
#             rho, theta = line[0]
#             a = np.cos(theta)
#             b = np.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             x1 = int(x0 + 1000 * (-b))
#             y1 = int(y0 + 1000 * (a))
#             x2 = int(x0 - 1000 * (-b))
#             y2 = int(y0 - 1000 * (a))
#             cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

#     # Detectar "patitos" con contornos
#     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > 500:  # Ajusta esto según tamaño de patitos
#             x, y, w, h = cv2.boundingRect(cnt)
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cx = x + w//2
#             cy = y + h//2
#             cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
#             cv2.putText(frame, f"({cx},{cy})", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

#     cv2.imshow('Grid + Patitos', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # --- Parámetros ajustables ---
# VIDEO_PATH = 'patitos.mp4'
# AREA_THRESHOLD = 500    # área mínima para considerar un contorno
# HOUGH_THRESH = 150      # umbral de HoughLines
# GRID_SQUARE_CM = 10.0   # tamaño real de cada cuadrado de la malla

# # --- Inicializaciones ---
# cap = cv2.VideoCapture(VIDEO_PATH)
# if not cap.isOpened():
#     print("Error al abrir el video.")
#     exit()

# pixel_per_cm = None      # se llenará tras detectar la malla
# centroids = []           # lista de (x, y) del patito en pixeles

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blur, 50, 150)

#     # 1) Detección de líneas Hough para estimar escala
#     if pixel_per_cm is None:
#         lines = cv2.HoughLines(edges, 1, np.pi / 180, HOUGH_THRESH)
#         if lines is not None:
#             # extraer solo líneas casi verticales
#             rhos = [l[0][0] for l in lines if abs(np.sin(l[0][1])) > 0.9]
#             # round y unique
#             rhos_uniq = sorted(set(int(round(r)) for r in rhos))
#             if len(rhos_uniq) >= 2:
#                 # diferencias entre líneas adyacentes
#                 diffs = np.diff(rhos_uniq)
#                 median_pix = np.median(diffs)
#                 pixel_per_cm = median_pix / GRID_SQUARE_CM
#                 print(f"Escala: {pixel_per_cm:.2f} pixeles/cm")
#             else:
#                 print("No hubo suficientes líneas verticales para calcular escala.")
    
#     # 2) Detección de contornos del patito (umbral inverso)
#     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # 3) Filtrar por área y elegir el más oscuro (patito negro)
#     candidatos = []
#     for cnt in contours:
#         if cv2.contourArea(cnt) < AREA_THRESHOLD:
#             continue
#         # máscara para medir brillo medio en gris
#         mask = np.zeros(gray.shape, np.uint8)
#         cv2.drawContours(mask, [cnt], -1, 255, -1)
#         mean_val = cv2.mean(gray, mask=mask)[0]
#         candidatos.append((cnt, mean_val))

#     if candidatos:
#         # escoger contorno con valor medio de gris más bajo (más oscuro)
#         cnt_obj, _ = min(candidatos, key=lambda x: x[1])
#         x, y, w, h = cv2.boundingRect(cnt_obj)
#         cx, cy = x + w//2, y + h//2
#         centroids.append((cx, cy))

#         # Dibujar cuadro y centro
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
#         cv2.putText(frame, f"Patito ({cx},{cy})", (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#     # 4) Mostrar resultado
#     cv2.imshow('Seguimiento Patito + Malla', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # 5) Cálculo de distancia recorrida en cm
# if pixel_per_cm is not None and len(centroids) >= 2:
#     # distancias entre fotogramas consecutivos (en pixeles)
#     dists_pix = [
#         np.hypot(centroids[i+1][0] - centroids[i][0],
#                  centroids[i+1][1] - centroids[i][1])
#         for i in range(len(centroids)-1)
#     ]
#     total_pix = sum(dists_pix)
#     total_cm = total_pix / pixel_per_cm
#     print(f"Distancia total recorrida por el patito: {total_cm:.2f} cm")
# else:
#     print("No se pudo calcular la distancia recorrida.")

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # --- Parámetros ajustables ---
# VIDEO_PATH = 'patitos.mp4'
# AREA_THRESHOLD = 500      # área mínima para considerar un contorno
# HOUGH_THRESH = 150        # umbral de HoughLines
# GRID_SQUARE_CM = 10.0     # tamaño real de cada cuadrado de la malla

# # --- Inicializaciones ---
# cap = cv2.VideoCapture(VIDEO_PATH)
# if not cap.isOpened():
#     print("Error al abrir el video.")
#     exit()

# # Para modelar el fondo y calcular contraste
# bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

# pixel_per_cm = None        # se llenará tras detectar la malla
# centroids = []             # lista de (x, y) del patito en pixeles

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blur, 50, 150)

#     # 1) Escala de pixeles/cm mediante líneas verticales de la malla
#     if pixel_per_cm is None:
#         lines = cv2.HoughLines(edges, 1, np.pi / 180, HOUGH_THRESH)
#         if lines is not None:
#             rhos = [l[0][0] for l in lines if abs(np.sin(l[0][1])) > 0.9]
#             rhos_uniq = sorted(set(int(round(r)) for r in rhos))
#             if len(rhos_uniq) >= 2:
#                 diffs = np.diff(rhos_uniq)
#                 median_pix = np.median(diffs)
#                 pixel_per_cm = median_pix / GRID_SQUARE_CM
#                 print(f"Escala: {pixel_per_cm:.2f} pixeles/cm")
#             else:
#                 print("No hubo suficientes líneas para escala.")

#     # 2) Máscara de movimiento para aislar objetos en primer plano
#     fg_mask = bg_subtractor.apply(frame)
#     fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

#     # 3) Umbral adaptativo para detectar regiones oscuras
#     thresh = cv2.adaptiveThreshold(
#         blur, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV,
#         blockSize=11, C=2
#     )
#     # Combina movimiento + umbral para focalizar patito
#     combined = cv2.bitwise_and(fg_mask, thresh)

#     contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     candidatos = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area < AREA_THRESHOLD:
#             continue

#         # Calcula brillo medio dentro del contorno
#         mask = np.zeros(gray.shape, np.uint8)
#         cv2.drawContours(mask, [cnt], -1, 255, -1)
#         mean_gray = cv2.mean(gray, mask=mask)[0]

#         # Calcula contraste medio respecto al fondo
#         bg_vals = cv2.mean(gray, mask=cv2.bitwise_not(mask))[0]
#         contraste = abs(bg_vals - mean_gray)

#         candidatos.append((cnt, mean_gray, contraste))

#     if candidatos:
#         # Ordena por (1) menor brillo, (2) mayor contraste
#         cnt_obj, _, _ = sorted(
#             candidatos,
#             key=lambda x: (x[1], -x[2])
#         )[0]
#         x, y, w, h = cv2.boundingRect(cnt_obj)
#         cx, cy = x + w//2, y + h//2
#         centroids.append((cx, cy))

#         # Dibujar detección
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
#         cv2.putText(frame, f"Patito", (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#     cv2.imshow('Detección Mejorada', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # --- Cálculo y gráfica de trayectoria en cm ---
# if pixel_per_cm is None or len(centroids) < 2:
#     print("No se pudo calcular trayectoria.")
# else:
#     traj_cm = [(x/pixel_per_cm, y/pixel_per_cm) for x, y in centroids]
#     xs, ys = zip(*traj_cm)

#     # Distancias totales
#     dists = [np.hypot(xs[i+1]-xs[i], ys[i+1]-ys[i])
#              for i in range(len(xs)-1)]
#     total_cm = sum(dists)
#     print(f"Distancia total recorrida: {total_cm:.2f} cm")

#     # Graficar
#     plt.figure(figsize=(6,6))
#     plt.plot(xs, ys, '-o', label='Trayectoria')
#     for i in range(len(xs)-1):
#         dx, dy = xs[i+1]-xs[i], ys[i+1]-ys[i]
#         plt.arrow(xs[i], ys[i], dx, dy, head_width=0.5, length_includes_head=True)
#     plt.gca().invert_yaxis()
#     plt.xlabel('X (cm)')
#     plt.ylabel('Y (cm)')
#     plt.title('Trayectoria y Dirección del Patito')
#     plt.grid(True)
#     plt.legend()
#     plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Parámetros ajustables ---
VIDEO_PATH = 'patitos.mp4'
AREA_THRESHOLD = 500      # área mínima para considerar un contorno
HOUGH_THRESH = 150        # umbral de HoughLines
GRID_SQUARE_CM = 10.0     # tamaño real de cada cuadrado de la malla

# --- Inicializaciones ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error al abrir el video.")
    exit()

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=50, detectShadows=True)

pixel_per_cm = None
centroids = []
prev_centroid = None  # posición del patito en el fotograma anterior

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 1) Calcular escala (pix/cm)
    if pixel_per_cm is None:
        lines = cv2.HoughLines(edges, 1, np.pi / 180, HOUGH_THRESH)
        if lines is not None:
            rhos = [l[0][0] for l in lines if abs(np.sin(l[0][1])) > 0.9]
            rhos_uniq = sorted(set(int(round(r)) for r in rhos))
            if len(rhos_uniq) >= 2:
                diffs = np.diff(rhos_uniq)
                median_pix = np.median(diffs)
                pixel_per_cm = median_pix / GRID_SQUARE_CM
                print(f"Escala: {pixel_per_cm:.2f} pixeles/cm")
    
    # 2) Máscara de movimiento y umbral adaptativo
    fg_mask = bg_subtractor.apply(frame)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2
    )
    combined = cv2.bitwise_and(fg_mask, thresh)
    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < AREA_THRESHOLD:
            continue

        # brillo medio dentro del contorno
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_gray = cv2.mean(gray, mask=mask)[0]
        # contraste con fondo
        bg_mean = cv2.mean(gray, mask=cv2.bitwise_not(mask))[0]
        contraste = abs(bg_mean - mean_gray)
        # centroid
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w//2, y + h//2

        candidatos.append({
            'cnt': cnt,
            'mean_gray': mean_gray,
            'contraste': contraste,
            'centroid': (cx, cy)
        })

    elegido = None
    if candidatos:
        if prev_centroid is None:
            # primer fotograma: igual que antes, por brillo+contraste
            elegido = min(
                candidatos,
                key=lambda c: (c['mean_gray'], -c['contraste'])
            )
        else:
            # fotogramas siguientes: elige el más cercano al prev_centroid,
            # penalizando también brillo/contraste para romper empates
            def score(c):
                dx = c['centroid'][0] - prev_centroid[0]
                dy = c['centroid'][1] - prev_centroid[1]
                dist = np.hypot(dx, dy)
                return dist - 0.01 * c['contraste']  # contraste reduce el "score"

            elegido = min(candidatos, key=score)

        cx, cy = elegido['centroid']
        centroids.append((cx, cy))
        prev_centroid = (cx, cy)

        # dibujar en el frame
        x, y, w, h = cv2.boundingRect(elegido['cnt'])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame, "Patito", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('Seguimiento Temporal', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- Gráfica final de trayectoria (en cm) ---
if pixel_per_cm and len(centroids) >= 2:
    traj_cm = [(x/pixel_per_cm, y/pixel_per_cm) for x, y in centroids]
    xs, ys = zip(*traj_cm)

    plt.figure(figsize=(6,6))
    plt.plot(xs, ys, '-o', label='Trayectoria')
    for i in range(len(xs)-1):
        dx, dy = xs[i+1]-xs[i], ys[i+1]-ys[i]
        plt.arrow(xs[i], ys[i], dx, dy,
                  head_width=0.5, length_includes_head=True)
    plt.gca().invert_yaxis()
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.title('Trayectoria y Dirección (seguimiento temporal)')
    plt.grid(True)
    plt.legend()
    plt.show()
else:
    print("No se pudo graficar trayectoria.")
