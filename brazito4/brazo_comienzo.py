import cv2
import mediapipe as mp
import numpy as np
import sys
import time
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

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

# Clases y funciones auxiliares
class JointController:
    """Clase para gestionar un joint específico y sus propiedades"""
    def __init__(self, name, sim, handle=None):
        self.name = name
        self.sim = sim
        self.handle = handle
        self.position = [0, 0, 0]
        self.orientation = [0, 0, 0, 1]  # Cuaternión [x, y, z, w]
        self.last_update_time = time.time()
        
    def get_handle(self):
        """Obtiene el handle del objeto desde CoppeliaSim"""
        if self.handle is None:
            try:
                self.handle = self.sim.getObject(self.name)
            except Exception as e:
                print(f"Error al obtener handle para {self.name}: {e}")
                return False
        return True
        
    def update_position(self, position):
        """Actualiza la posición del objeto en CoppeliaSim"""
        if self.handle is not None:
            self.position = position
            self.sim.setObjectPosition(self.handle, -1, position)
            self.last_update_time = time.time()
            
    def update_orientation(self, orientation):
        """Actualiza la orientación del objeto en CoppeliaSim usando cuaterniones"""
        if self.handle is not None:
            self.orientation = orientation
            self.sim.setObjectQuaternion(self.handle, -1, orientation)
            
    def get_position(self):
        """Obtiene la posición actual del objeto en CoppeliaSim"""
        if self.handle is not None:
            self.position = self.sim.getObjectPosition(self.handle, -1)
            return self.position
        return None

# Clase para gestionar objetos físicos (con propiedades dinámicas)
class PhysicalObject:
    """Clase para gestionar objetos físicos en CoppeliaSim"""
    def __init__(self, name, sim, handle=None):
        self.name = name
        self.sim = sim
        self.handle = handle
        self.is_dynamic = False
        
    def get_handle(self):
        """Obtiene el handle del objeto desde CoppeliaSim"""
        if self.handle is None:
            try:
                self.handle = self.sim.getObject(self.name)
            except Exception as e:
                print(f"Error al obtener handle para {self.name}: {e}")
                return False
        return True
    
    def enable_dynamics(self, enable=True):
        """Habilita o deshabilita las propiedades dinámicas del objeto"""
        if self.handle is not None:
            # Configurar como dinámico (afectado por física)
            self.sim.setObjectInt32Param(self.handle, self.sim.objintparam_static, 0 if enable else 1)
            
            # Establecer propiedades de material (fricción, etc.)
            if enable:
                # Establecer material: responde a colisiones
                self.sim.setShapeMassAndInertia(self.handle, 0.1, [0,0,0], [1,1,1])
            
            self.is_dynamic = enable
            return True
        return False

# Mapear coordenadas de cámara a espacio 3D de CoppeliaSim
def map_coords_to_sim(point_2d, depth=0.5, w=640, h=480, scale_factor=0.5):
    """
    Transforma coordenadas 2D de la cámara a espacio 3D de CoppeliaSim
    """
    # Mapear de coordenadas de imagen [0,w]x[0,h] a [-1,1]x[-1,1]
    x = (point_2d[0] / w - 0.5) * 2 * scale_factor
    y = -(point_2d[1] / h - 0.5) * 2 * scale_factor
    z = depth  # O -depth, según la orientación de tu sim
    
    return [x, y, z]

# Función para calcular orientación basada en dos puntos (vector dirección)
def calculate_orientation(point1, point2):
    """
    Calcula un cuaternión que representa la orientación desde point1 a point2
    Args:
        point1: Punto origen [x,y,z]
        point2: Punto destino [x,y,z]
    Returns:
        Cuaternión [x,y,z,w] que representa la rotación
    """
    # Calcular vector dirección normalizado
    direction = np.array(point2) - np.array(point1)
    if np.linalg.norm(direction) < 1e-6:
        return [0, 0, 0, 1]  # Cuaternión identidad si los puntos están muy cerca
        
    direction = direction / np.linalg.norm(direction)
    
    # Vector referencia (asumimos que queremos alinear con eje Z positivo)
    ref_vector = np.array([0, 0, 1])
    
    # Si los vectores son casi paralelos, usar un cuaternión identidad
    if np.abs(np.dot(direction, ref_vector)) > 0.99999:
        if np.dot(direction, ref_vector) > 0:
            return [0, 0, 0, 1]  # Identidad
        else:
            return [1, 0, 0, 0]  # Rotación 180° alrededor de X
    
    # Calcular eje de rotación (perpendicular a ambos vectores)
    axis = np.cross(ref_vector, direction)
    axis = axis / np.linalg.norm(axis)
    
    # Calcular ángulo entre vectores
    angle = math.acos(np.clip(np.dot(ref_vector, direction), -1.0, 1.0))
    
    # Crear cuaternión [x,y,z,w]
    sin_half_angle = math.sin(angle / 2)
    qx = axis[0] * sin_half_angle
    qy = axis[1] * sin_half_angle
    qz = axis[2] * sin_half_angle
    qw = math.cos(angle / 2)
    
    return [qx, qy, qz, qw]

# Función para actualizar la apertura de la pinza en CoppeliaSim
def update_gripper_state(sim, distance, controllers, min_dist=20, max_dist=120):
    try:
        # Normalizar y recortar para obtener valor entre 0 (cerrado) y 1 (abierto)
        normalized_distance = np.clip((distance - min_dist) / (max_dist - min_dist), 0.0, 1.0)
        sim.setFloatSignal('gripper_opening', float(normalized_distance))
        return normalized_distance  # Devuelve el valor para mostrarlo en pantalla
    except Exception as e:
        print(f"Error al actualizar pinza: {e}")
        return None

# Calcular distancia euclidiana entre dos puntos
def distance(p1, p2):
    """Calcula distancia euclidiana entre dos puntos"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Función para calcular la profundidad basada en la distancia entre hombro y muñeca
def estimate_depth(shoulder_pos, wrist_pos, min_depth=0.1, max_depth=0.5):
    """
    Estima la profundidad (coordenada z) basada en la distancia entre hombro y muñeca
    """
    arm_length = distance(shoulder_pos, wrist_pos)
    
    # Ajustar estos valores según tu cámara y espacio de trabajo
    max_arm_length = 400  # Aumentado para mejor detección
    min_arm_length = 50   # Reducido para mejor detección
    
    # Calcular profundidad normalizada e invertida
    norm_length = np.clip((arm_length - min_arm_length) / (max_arm_length - min_arm_length), 0.0, 1.0)
    depth = min_depth + (1.0 - norm_length) * (max_depth - min_depth)
    
    return depth

# Configurar cinemática inversa para el brazo robótico
def setup_inverse_kinematics(sim, target_handle, tip_handle, base_handle):
    """
    Configura la cinemática inversa para el brazo robótico
    """
    try:
        # Verificar que los handles son válidos
        if not isinstance(target_handle, int) or not isinstance(tip_handle, int) or not isinstance(base_handle, int):
            print(f"Error: Handles inválidos - target: {type(target_handle)}, tip: {type(tip_handle)}, base: {type(base_handle)}")
            return None

        # Crear un nuevo grupo de IK
        ikGroupHandle = sim.createIkGroup(0)  # 0 = método DLS
        if ikGroupHandle == -1:
            print("Error al crear grupo IK")
            return None
            
        print(f"Grupo IK creado: {ikGroupHandle}")
        
        # Crear el elemento de IK (tip -> target)
        # Asegurarse de que los handles son enteros
        ikElementHandle = sim.createIkElement(
            int(ikGroupHandle),
            int(tip_handle),
            int(target_handle),
            int(base_handle)
        )
        
        if ikElementHandle == -1:
            print("Error al crear elemento IK")
            return None
            
        print(f"Cinemática inversa configurada: Grupo={ikGroupHandle}, Elemento={ikElementHandle}")
        return ikGroupHandle
        
    except Exception as e:
        print(f"Error al configurar cinemática inversa: {e}")
        print(f"Handles recibidos - target: {target_handle}, tip: {tip_handle}, base: {base_handle}")
        return None

def solve_inverse_kinematics(sim, ikGroupHandle):
    """
    Resuelve la cinemática inversa para la posición actual del target
    """
    if ikGroupHandle is not None:
        try:
            # Llamar al solucionador de IK
            sim.handleIkGroup(ikGroupHandle)
        except Exception as e:
            print(f"Error al resolver IK: {e}")

# Función para crear un objeto cúbico físico
def create_dynamic_cube(sim, size=0.05, position=[0,0,0], color=[1,0,0]):
    """
    Crea un cubo con propiedades dinámicas en CoppeliaSim
    Args:
        sim: Objeto simulación de CoppeliaSim
        size: Tamaño del cubo
        position: Posición inicial [x,y,z]
        color: Color [r,g,b] en rango 0-1
    Returns:
        Handle del objeto creado o None si falla
    """
    try:
        # Crear cubo
        handle = sim.createPrimitiveShape(sim.primitiveshape_cube, [size, size, size], 0, position)
        sim.setObjectColor(handle, 0, sim.colorcomponent_ambient_diffuse, color)
        sim.setObjectInt32Param(handle, sim.objintparam_static, 0)  # Dinámico
        sim.setShapeMassAndInertia(handle, 0.1, [0,0,0], [1,1,1])
        return handle
    except Exception as e:
        print(f"Error al crear cubo dinámico: {e}")
        return None

# Función para inicializar y obtener handles de objetos importantes
def initialize_joint_controllers(sim):
    """
    Inicializa controladores para los joints y dummies relevantes del NiryoOne
    Args:
        sim: Objeto simulación de CoppeliaSim
    Returns:
        Diccionario con los controladores de joints
    """
    joint_names = [
        '/NiryoOne/NiryoOne_target',    # Target para el efector final (dummy)
        '/NiryoOne/NiryoOne_tip',       # Tip del efector final (dummy)
        '/NiryoOne/NiryoOne_joint1',     # Joint 1
        '/NiryoOne/NiryoOne_joint2',     # Joint 2
        '/NiryoOne/NiryoOne_joint3',     # Joint 3
        '/NiryoOne/NiryoOne_joint4',     # Joint 4
        '/NiryoOne/NiryoOne_joint5',     # Joint 5
        '/NiryoOne/NiryoOne_joint6',     # Joint 6
        '/NiryoOne/NiryoOneGripper'     # Pinza (si existe)
    ]
    controllers = {}
    for name in joint_names:
        controller = JointController(name, sim)
        if controller.get_handle():
            controllers[name] = controller
            print(f"Joint '{name}' inicializado correctamente")
        else:
            print(f"¡ADVERTENCIA! No se pudo obtener handle para '{name}'")
    return controllers

# Función principal
def main():
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        sys.exit(1)

    # Intentar leer un frame para obtener dimensiones
    ret, frame = cap.read()
    if not ret:
        print("Error al leer frame de la cámara")
        cap.release()
        sys.exit(1)
    
    frame_height, frame_width, _ = frame.shape
    print(f"Resolución de cámara: {frame_width}x{frame_height}")

    # Inicializar cliente y simulación
    client = RemoteAPIClient()
    sim = client.require('sim')

    # Inicializar controladores de joints
    try:
        controllers = initialize_joint_controllers(sim)
        if not controllers:
            print("No se pudieron inicializar los controladores de joints, saliendo...")
            cap.release()
            sys.exit(1)

        # Crear el grupo IK solo una vez
        ikGroupHandle = None
        if '/NiryoOne/NiryoOne_target' in controllers and '/NiryoOne/NiryoOne_tip' in controllers and '/NiryoOne/NiryoOne_joint1' in controllers:
            target_handle = controllers['/NiryoOne/NiryoOne_target'].handle
            tip_handle = controllers['/NiryoOne/NiryoOne_tip'].handle
            base_handle = controllers['/NiryoOne/NiryoOne_joint1'].handle
            print(f"Configurando IK con handles - target: {target_handle}, tip: {tip_handle}, base: {base_handle}")
            ikGroupHandle = setup_inverse_kinematics(
                sim,
                target_handle,
                tip_handle,
                base_handle
            )

        # Al inicio, después de obtener el handle del target:
        initial_target_pos = None
        if '/NiryoOne/NiryoOne_target' in controllers:
            initial_target_pos = controllers['/NiryoOne/NiryoOne_target'].get_position()

    except Exception as e:
        print(f"Error al inicializar controladores: {e}")
        cap.release()
        sys.exit(1)

    # Variables para tracking
    smoothing_factor = 0.7  # Factor para suavizado (0-1, mayor = más suave)
    last_positions = {
        'index': None,
        'thumb': None,
        'wrist': None,
        'elbow': None,
        'shoulder': None
    }
    
    hand_detected = False
    last_hand_detected_time = 0
    hand_timeout = 1.0  # segundos para considerar mano perdida
    
    # Variables para control de objetos dinámicos
    last_cube_creation_time = 0
    cube_creation_timeout = 3.0  # segundos mínimos entre creaciones
    created_cubes = []  # Lista para almacenar handles de cubos creados

    print("Iniciando captura de video y transmisión a CoppeliaSim...")
    print("Presiona 'q' para salir, 'c' para crear un cubo dinámico")

    try:
        wrist_pos_3d = None
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar frame")
                break

            frame = cv2.flip(frame, 1)  # Espejo horizontal para facilitar uso
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesar pose para codo y muñeca
            pose_results = pose.process(frame_rgb)
            
            # Procesar mano
            hands_results = hands.process(frame_rgb)

            # Variables para almacenar posiciones de los puntos clave
            shoulder_pos = None
            elbow_pos = None
            wrist_pos = None
            index_pos = None
            thumb_pos = None
            
            # Variable para la profundidad estimada (coordenada Z)
            estimated_depth = 0.5  # Valor por defecto

            # Extraer puntos del brazo (hombro, codo y muñeca)
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                
                # Usar lado derecho por defecto (landmarks 12, 14, 16)
                # Podríamos implementar detección automática del lado más visible
                
                # Hombro derecho
                if landmarks[12].visibility > 0.5:
                    hombro = np.array([
                        int(landmarks[12].x * frame_width), 
                        int(landmarks[12].y * frame_height)
                    ])
                    shoulder_pos = hombro
                    cv2.circle(frame, tuple(hombro), 8, (0, 255, 0), cv2.FILLED)
                
                # Codo derecho
                if landmarks[14].visibility > 0.5:
                    codo = np.array([
                        int(landmarks[14].x * frame_width), 
                        int(landmarks[14].y * frame_height)
                    ])
                    elbow_pos = codo
                    cv2.circle(frame, tuple(codo), 8, (0, 255, 0), cv2.FILLED)
                
                # Muñeca derecha
                if landmarks[16].visibility > 0.5:
                    muneca = np.array([
                        int(landmarks[16].x * frame_width), 
                        int(landmarks[16].y * frame_height)
                    ])
                    wrist_pos = muneca
                    cv2.circle(frame, tuple(muneca), 8, (0, 255, 0), cv2.FILLED)
                
                # Dibujar líneas entre puntos visibles
                if shoulder_pos is not None and elbow_pos is not None:
                    cv2.line(frame, tuple(shoulder_pos), tuple(elbow_pos), (0, 255, 0), 3)
                
                if elbow_pos is not None and wrist_pos is not None:
                    cv2.line(frame, tuple(elbow_pos), tuple(wrist_pos), (0, 255, 0), 3)
                    
                # Estimar profundidad basada en la distancia entre hombro y muñeca
                if shoulder_pos is not None and wrist_pos is not None:
                    estimated_depth = estimate_depth(shoulder_pos, wrist_pos)
                    cv2.putText(frame, f"Depth: {estimated_depth:.2f}", 
                              (10, frame_height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                # Mapear a coordenadas 3D para CoppeliaSim
                if shoulder_pos is not None:
                    shoulder_pos_3d = map_coords_to_sim(shoulder_pos, estimated_depth + 0.1, frame_width, frame_height)
                    if last_positions['shoulder'] is not None:
                        shoulder_pos_3d = [
                            last_positions['shoulder'][i] * smoothing_factor + 
                            shoulder_pos_3d[i] * (1 - smoothing_factor) for i in range(3)
                        ]
                    last_positions['shoulder'] = shoulder_pos_3d
                    if '/NiryoOne/NiryoOne_target' in controllers:
                        controllers['/NiryoOne/NiryoOne_target'].update_position(shoulder_pos_3d)
                
                if elbow_pos is not None:
                    elbow_pos_3d = map_coords_to_sim(elbow_pos, estimated_depth, frame_width, frame_height)
                    if last_positions['elbow'] is not None:
                        elbow_pos_3d = [
                            last_positions['elbow'][i] * smoothing_factor + 
                            elbow_pos_3d[i] * (1 - smoothing_factor) for i in range(3)
                        ]
                    last_positions['elbow'] = elbow_pos_3d
                    if '/NiryoOne/NiryoOne_target' in controllers:
                        controllers['/NiryoOne/NiryoOne_target'].update_position(elbow_pos_3d)
                
                if wrist_pos is not None:
                    wrist_pos_3d = map_coords_to_sim(wrist_pos, estimated_depth, frame_width, frame_height)
                    if last_positions['wrist'] is not None:
                        wrist_pos_3d = [
                            last_positions['wrist'][i] * smoothing_factor +
                            wrist_pos_3d[i] * (1 - smoothing_factor) for i in range(3)
                        ]
                    last_positions['wrist'] = wrist_pos_3d
                    if '/NiryoOne/NiryoOne_target' in controllers:
                        controllers['/NiryoOne/NiryoOne_target'].update_position(wrist_pos_3d)
                        # También actualizar la orientación del target
                        if elbow_pos is not None:
                            elbow_pos_3d = map_coords_to_sim(elbow_pos, estimated_depth, frame_width, frame_height)
                            orientation = calculate_orientation(wrist_pos_3d, elbow_pos_3d)
                            controllers['/NiryoOne/NiryoOne_target'].update_orientation(orientation)
                
                # Resolver IK solo si el grupo fue creado correctamente
                if ikGroupHandle is not None:
                    solve_inverse_kinematics(sim, ikGroupHandle)

                if wrist_pos_3d is not None:
                    print(f"Target 3D: {wrist_pos_3d}")

            # Procesar mano para obtener dedos
            current_time = time.time()
            if hands_results.multi_hand_landmarks:
                hand_detected = True
                last_hand_detected_time = current_time
                
                hand_landmarks = hands_results.multi_hand_landmarks[0]
                
                # Dibujar todos los puntos de la mano
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(120, 160, 240), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 220, 20), thickness=2)
                )
                
                # Obtener posiciones de puntos clave
                landmarks = hand_landmarks.landmark
                
                # Dedo índice (punta)
                index_tip = landmarks[8]
                index_pos = np.array([
                    int(index_tip.x * frame_width),
                    int(index_tip.y * frame_height)
                ])
                
                # Pulgar (punta)
                thumb_tip = landmarks[4]
                thumb_pos = np.array([
                    int(thumb_tip.x * frame_width),
                    int(thumb_tip.y * frame_height)
                ])
                
                # Resaltar puntos clave
                cv2.circle(frame, tuple(index_pos), 10, (0, 255, 255), cv2.FILLED)
                cv2.circle(frame, tuple(thumb_pos), 10, (0, 255, 255), cv2.FILLED)
                
                # Calcular distancia entre dedos
                finger_distance = distance(thumb_pos, index_pos)
                
                # Actualizar pinza con la distancia entre dedos
                gripper_signal = update_gripper_state(sim, finger_distance, controllers, min_dist=20, max_dist=120)
                
                # Mostrar información de debug en la pantalla
                if gripper_signal is not None:
                    cv2.putText(frame, f"Dist: {finger_distance:.1f}px | Gripper: {gripper_signal:.2f}", 
                                (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, f"Dist: {finger_distance:.1f}px", 
                                (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Depth: {estimated_depth:.2f}", 
                            (10, frame_height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Dibujar línea entre dedos
                cv2.line(frame, tuple(thumb_pos), tuple(index_pos), (0, 255, 255), 2)
                
                # Mapear a coordenadas 3D para CoppeliaSim
                index_pos_3d = map_coords_to_sim(index_pos, estimated_depth - 0.15, frame_width, frame_height)
                thumb_pos_3d = map_coords_to_sim(thumb_pos, estimated_depth - 0.15, frame_width, frame_height)
                
                # Aplicar suavizado
                if last_positions['index'] is not None:
                    index_pos_3d = [
                        last_positions['index'][i] * smoothing_factor + 
                        index_pos_3d[i] * (1 - smoothing_factor) for i in range(3)
                    ]
                
                if last_positions['thumb'] is not None:
                    thumb_pos_3d = [
                        last_positions['thumb'][i] * smoothing_factor + 
                        thumb_pos_3d[i] * (1 - smoothing_factor) for i in range(3)
                    ]
                
                # Actualizar posiciones en CoppeliaSim
                if '/NiryoOne/NiryoOne_target' in controllers:
                    controllers['/NiryoOne/NiryoOne_target'].update_position(index_pos_3d)
                
                # Guardar posiciones para suavizado
                last_positions['index'] = index_pos_3d
                last_positions['thumb'] = thumb_pos_3d

            # Mostrar frame
            cv2.imshow('Frame', frame)
            
            # Manejo de teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                current_time = time.time()
                if current_time - last_cube_creation_time > cube_creation_timeout:
                    # Crear cubo en la posición actual del target
                    if '/NiryoOne/NiryoOne_target' in controllers:
                        target_pos = controllers['/NiryoOne/NiryoOne_target'].get_position()
                        if target_pos is not None:
                            cube_handle = create_dynamic_cube(
                                sim,
                                size=0.05,
                                position=target_pos,
                                color=[1, 0, 0]  # Rojo
                            )
                            if cube_handle is not None:
                                created_cubes.append(cube_handle)
                                last_cube_creation_time = current_time
                                print(f"Cubo creado en posición: {target_pos}")

            # Nueva función de mapeo:
            def map_height_to_sim_z(y_pixel, h=480, z_min=0.1, z_max=0.4):
                """
                Mapea la altura de la mano en la imagen (y_pixel) al eje Z del sim.
                y_pixel: coordenada vertical en la imagen (0 arriba, h abajo)
                z_min: altura mínima en el sim
                z_max: altura máxima en el sim
                """
                # Invertir eje: 0 (arriba) -> z_max, h (abajo) -> z_min
                norm = 1.0 - (y_pixel / h)
                z = z_min + norm * (z_max - z_min)
                return z

            # En el bucle principal, al actualizar la posición del target:
            if wrist_pos is not None and initial_target_pos is not None:
                # Solo actualiza el eje Z
                z = map_height_to_sim_z(wrist_pos[1], frame_height, z_min=0.1, z_max=0.4)
                new_target_pos = [initial_target_pos[0], initial_target_pos[1], z]
                controllers['/NiryoOne/NiryoOne_target'].update_position(new_target_pos)

    except Exception as e:
        print(f"Error durante la ejecución: {e}")
    finally:
        # Limpiar recursos
        cap.release()
        cv2.destroyAllWindows()
        print("Programa terminado")

if __name__ == "__main__":
    main()