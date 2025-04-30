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

    # --- 1) Detección de líneas con Hough y dibujo ---
    lines = cv2.HoughLines(edges, 1, np.pi / 180, HOUGH_THRESH)
    if lines is not None:
        # Dibujar todas las líneas detectadas en azul
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # puntos extremos de la línea
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * ( a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * ( a))
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Estimación de escala (pix/cm) la primera vez
        if pixel_per_cm is None:
            # Conservamos solo líneas casi verticales para medir distancias horizontales
            rhos = [l[0][0] for l in lines if abs(np.sin(l[0][1])) > 0.9]
            if len(rhos) >= 2:
                rhos_uniq = sorted(set(int(round(r)) for r in rhos))
                if len(rhos_uniq) >= 2:
                    diffs = np.diff(rhos_uniq)
                    median_pix = np.median(diffs)
                    pixel_per_cm = median_pix / GRID_SQUARE_CM
                    print(f"Escala: {pixel_per_cm:.2f} pixeles/cm")

    # --- 2) Máscara de movimiento y umbral adaptativo ---
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
            # primer fotograma: por brillo + contraste
            elegido = min(
                candidatos,
                key=lambda c: (c['mean_gray'], -c['contraste'])
            )
        else:
            # fotogramas siguientes: más cercano al prev_centroid
            def score(c):
                dx = c['centroid'][0] - prev_centroid[0]
                dy = c['centroid'][1] - prev_centroid[1]
                dist = np.hypot(dx, dy)
                return dist - 0.01 * c['contraste']

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

    cv2.imshow('Seguimiento Temporal + HoughLines', frame)
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
