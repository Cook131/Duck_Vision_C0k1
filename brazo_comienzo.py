import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Pose para codo y muñeca
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Inicializar MediaPipe Hands para la mano
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Variables para el cuadro rojo
box_pos = np.array([300, 200], dtype=np.int32)
box_size = 150
dragging = False
drag_threshold = 20  # distancia para detectar cercanía al perímetro
release_distance_threshold = 150  # distancia entre dedos para soltar la caja

# Para mantener la diferencia relativa entre dedos y cuadro al iniciar drag
offset_index = np.array([0, 0], dtype=np.int32)
offset_thumb = np.array([0, 0], dtype=np.int32)

# Distancia base pulgar-indice al iniciar agarre (para detectar zoom claro)
base_dist_pulgar_indice = None
zoom_threshold = 15  # pixeles mínimos para considerar zoom

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def point_to_rect_distance(point, rect_tl, rect_br):
    """
    Calcula la distancia mínima desde un punto a un rectángulo definido por top-left y bottom-right.
    Si el punto está dentro del rectángulo, la distancia es 0.
    """
    px, py = point
    left, top = rect_tl
    right, bottom = rect_br

    if left <= px <= right and top <= py <= bottom:
        return 0  # Dentro del rectángulo

    dx = max(left - px, 0, px - right)
    dy = max(top - py, 0, py - bottom)
    return np.sqrt(dx*dx + dy*dy)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar pose para codo y muñeca
    pose_results = pose.process(frame_rgb)
    # Procesar mano
    hands_results = hands.process(frame_rgb)

    # Extraer puntos relevantes del brazo derecho (codo y muñeca)
    if pose_results.pose_landmarks:
        pose_landmarks = pose_results.pose_landmarks.landmark
        codo = np.array([int(pose_landmarks[13].x * w), int(pose_landmarks[13].y * h)])
        muneca_pose = np.array([int(pose_landmarks[15].x * w), int(pose_landmarks[15].y * h)])

        cv2.circle(frame, tuple(codo), 8, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, tuple(muneca_pose), 8, (0, 255, 0), cv2.FILLED)
        cv2.line(frame, tuple(codo), tuple(muneca_pose), (0, 255, 0), 3)
    else:
        codo = None
        muneca_pose = None

    if hands_results.multi_hand_landmarks:
        hand_landmarks = hands_results.multi_hand_landmarks[0]
        hand_points = {}
        idxs = [0, 1, 2, 5, 6, 9, 10, 13, 14, 17, 18]
        for idx in idxs:
            lm = hand_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            hand_points[idx] = (x, y)
            cv2.circle(frame, (x, y), 7, (255, 255, 0), cv2.FILLED)

        # Dibujar aristas mano
        for finger_base in [1, 5, 9, 13, 17]:
            if 0 in hand_points and finger_base in hand_points:
                cv2.line(frame, hand_points[0], hand_points[finger_base], (255, 255, 0), 3)
        finger_pairs = [(1, 2), (5, 6), (9, 10), (13, 14), (17, 18)]
        for start, end in finger_pairs:
            if start in hand_points and end in hand_points:
                cv2.line(frame, hand_points[start], hand_points[end], (255, 255, 0), 3)

        lm_index_tip = hand_landmarks.landmark[8]
        dedo_indice = np.array([int(lm_index_tip.x * w), int(lm_index_tip.y * h)])
        lm_thumb_tip = hand_landmarks.landmark[4]
        pulgar = np.array([int(lm_thumb_tip.x * w), int(lm_thumb_tip.y * h)])

        cv2.circle(frame, tuple(dedo_indice), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(frame, tuple(pulgar), 10, (0, 255, 255), cv2.FILLED)

        top_left = box_pos
        bottom_right = box_pos + box_size

        dist_index = point_to_rect_distance(dedo_indice, top_left, bottom_right)
        dist_thumb = point_to_rect_distance(pulgar, top_left, bottom_right)

        both_touching = (dist_index < drag_threshold) and (dist_thumb < drag_threshold)

        dist_between_fingers = distance(pulgar, dedo_indice)

        if both_touching and not dragging:
            dragging = True
            offset_index = dedo_indice - box_pos
            offset_thumb = pulgar - box_pos
            base_dist_pulgar_indice = dist_between_fingers

        if dragging:
            # Si los dedos se alejan demasiado, soltamos la caja
            if dist_between_fingers > release_distance_threshold:
                dragging = False
                base_dist_pulgar_indice = None
            else:
                # Mover caja manteniendo posición relativa promedio
                pos_index = dedo_indice - offset_index
                pos_thumb = pulgar - offset_thumb
                new_pos = ((pos_index + pos_thumb) / 2).astype(np.int32)

                new_pos[0] = np.clip(new_pos[0], 0, w - box_size)
                new_pos[1] = np.clip(new_pos[1], 0, h - box_size)
                box_pos = new_pos

                # Escalar solo si el zoom es claro
                diff_dist = dist_between_fingers - base_dist_pulgar_indice
                if abs(diff_dist) > zoom_threshold:
                    new_size = int(box_size + diff_dist)
                    box_size = np.clip(new_size, 50, 400)
                    base_dist_pulgar_indice = dist_between_fingers

        else:
            dragging = False
            base_dist_pulgar_indice = None

    else:
        dragging = False
        base_dist_pulgar_indice = None

    # Dibujar cuadro rojo
    cv2.rectangle(frame, tuple(box_pos), tuple(box_pos + box_size), (0, 0, 255), 3)

    # Instrucciones
    cv2.putText(frame, "Agarra cuadro con dedo indice y pulgar tocando borde", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Separa pulgar e indice para cambiar tamano mientras agarras", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Separa mucho los dedos para soltar", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Presiona 'q' para salir", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Caja pegada a dedos - Soltar con distancia", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose.close()
hands.close()
cap.release()
cv2.destroyAllWindows()
