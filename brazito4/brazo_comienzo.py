import cv2
import mediapipe as mp
import numpy as np
import sys
import time
import math
import os

# Añadir ruta de la API de CoppeliaSim para Mac
# Ajusta esta ruta según donde tengas instalado CoppeliaSim
COPPELIASIM_DIR = '/Applications/CoppeliaSim.app/Contents/MacOS/'
API_DIR = os.path.join(COPPELIASIM_DIR, 'programming/remoteApiBindings/python/python')

# Añadir la ruta al PYTHONPATH
if os.path.exists(API_DIR):
    sys.path.append(API_DIR)
else:
    print(f"¡ATENCIÓN! No se encuentra la API de CoppeliaSim en: {API_DIR}")
    print("Por favor verifica la ruta de instalación de CoppeliaSim")
    print("Buscando en rutas alternativas...")
    
    # Intentar encontrar la API en otras ubicaciones comunes
    alt_paths = [
        './remoteApi',  # Directorio local
        '~/Downloads/CoppeliaSim.app/Contents/MacOS/programming/remoteApiBindings/python/python',
        '/Applications/CoppeliaSim_Edu.app/Contents/MacOS/programming/remoteApiBindings/python/python'
    ]
    
    for path in alt_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            sys.path.append(expanded_path)
            print(f"API encontrada en: {expanded_path}")
            API_DIR = expanded_path
            break

# Importar API de CoppeliaSim
try:
    import sim
except Exception as e:
    print(f'Error importando módulo sim: {e}')
    print('Asegúrate de tener la API de CoppeliaSim para Python correctamente instalada')
    print('Para Mac, normalmente se encuentra en: /Applications/CoppeliaSim.app/Contents/MacOS/')
    print('Puedes copiar los archivos sim.py, simConst.py y remoteApi.dylib al directorio de este script')
    sys.exit(1)

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
    def __init__(self, name, clientID, handle=None):
        self.name = name
        self.clientID = clientID
        self.handle = handle
        self.position = [0, 0, 0]
        self.orientation = [0, 0, 0, 1]  # Cuaternión [x, y, z, w]
        self.last_update_time = time.time()
        
    def get_handle(self):
        """Obtiene el handle del objeto desde CoppeliaSim"""
        if self.handle is None:
            ret, self.handle = sim.simxGetObjectHandle(
                self.clientID, self.name, sim.simx_opmode_blocking)
            if ret != 0:
                print(f"Error al obtener handle para {self.name}: {ret}")
                return False
        return True
        
    def update_position(self, position, mode=sim.simx_opmode_oneshot):
        """Actualiza la posición del objeto en CoppeliaSim"""
        if self.handle is not None:
            self.position = position
            sim.simxSetObjectPosition(
                self.clientID, self.handle, -1, position, mode)
            self.last_update_time = time.time()
            
    def update_orientation(self, orientation, mode=sim.simx_opmode_oneshot):
        """Actualiza la orientación del objeto en CoppeliaSim usando cuaterniones"""
        if self.handle is not None:
            self.orientation = orientation
            sim.simxSetObjectQuaternion(
                self.clientID, self.handle, -1, orientation, mode)
            
    def get_position(self, mode=sim.simx_opmode_blocking):
        """Obtiene la posición actual del objeto en CoppeliaSim"""
        if self.handle is not None:
            ret, position = sim.simxGetObjectPosition(
                self.clientID, self.handle, -1, mode)
            if ret == 0:
                self.position = position
            return self.position
        return None

# Clase para gestionar objetos físicos (con propiedades dinámicas)
class PhysicalObject:
    """Clase para gestionar objetos físicos en CoppeliaSim"""
    def __init__(self, name, clientID, handle=None):
        self.name = name
        self.clientID = clientID
        self.handle = handle
        self.is_dynamic = False
        
    def get_handle(self):
        """Obtiene el handle del objeto desde CoppeliaSim"""
        if self.handle is None:
            ret, self.handle = sim.simxGetObjectHandle(
                self.clientID, self.name, sim.simx_opmode_blocking)
            if ret != 0:
                print(f"Error al obtener handle para {self.name}: {ret}")
                return False
        return True
    
    def enable_dynamics(self, enable=True):
        """Habilita o deshabilita las propiedades dinámicas del objeto"""
        if self.handle is not None:
            # Configurar como dinámico (afectado por física)
            sim.simxSetObjectIntParameter(
                self.clientID, self.handle, 
                sim.sim_shapeintparam_static, 0 if enable else 1, 
                sim.simx_opmode_oneshot)
            
            # Establecer propiedades de material (fricción, etc.)
            if enable:
                # Establecer material: responde a colisiones
                sim.simxSetObjectIntParameter(
                    self.clientID, self.handle,
                    sim.sim_shapefloatparam_mass, 0.5,  # Masa
                    sim.simx_opmode_oneshot)
                
                # Responde a la gravedad
                sim.simxSetObjectIntParameter(
                    self.clientID, self.handle,
                    sim.sim_objectspecialproperty_respondable, 1,
                    sim.simx_opmode_oneshot)
            
            self.is_dynamic = enable
            return True
        return False

# Función para conectar con CoppeliaSim
def connect_to_coppeliasim(port=19997):
    print('Conectando a CoppeliaSim...')
    sim.simxFinish(-1)  # Cerrar todas las conexiones abiertas
    clientID = sim.simxStart('127.0.0.1', port, True, True, 5000, 5)
    
    if clientID != -1:
        print('Conexión a CoppeliaSim establecida')
        # Verificar que la simulación está en ejecución
        sim_running = sim.simxGetInMessageInfo(clientID, sim.simx_headeroffset_server_state)
        if sim_running[0] & 1 == 0:
            print("ADVERTENCIA: La simulación en CoppeliaSim no está en ejecución.")
            print("Inicia la simulación con el botón Play para que la comunicación funcione correctamente.")
    else:
        print('Error al conectar con CoppeliaSim')
        print('Asegúrate de que CoppeliaSim está en ejecución y en modo servidor')
        print('Y que el puerto 19997 está habilitado para Remote API')
        print('Puedes habilitarlo en Tools > Options > Remote API')
        
    return clientID

# Mapear coordenadas de cámara a espacio 3D de CoppeliaSim
def map_coords_to_sim(point_2d, depth=0.5, w=640, h=480, scale_factor=0.5):
    """
    Transforma coordenadas 2D de la cámara a espacio 3D de CoppeliaSim
    Args:
        point_2d: Punto [x,y] en coordenadas de la cámara
        depth: Profundidad estimada (z) para el punto
        w, h: Dimensiones del frame de la cámara
        scale_factor: Factor de escala para ajustar el mapeo al espacio de trabajo
    Returns:
        Lista [x,y,z] con coordenadas en el espacio de CoppeliaSim
    """
    # Mapear de coordenadas de imagen [0,w]x[0,h] a [-1,1]x[-1,1]
    x = (point_2d[0] / w - 0.5) * 2
    y = -(point_2d[1] / h - 0.5) * 2  # Invertir eje Y (en imagen hacia abajo, en sim hacia arriba)
    
    # Aplicar factor de escala
    x *= scale_factor
    y *= scale_factor
    
    # Ajustar profundidad según lo que necesites en tu modelo
    z = depth
    
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
def update_gripper_state(clientID, distance, min_dist=20, max_dist=120):
    """
    Actualiza la apertura de la pinza basándose en la distancia entre dedos
    Args:
        clientID: ID de conexión a CoppeliaSim
        distance: Distancia entre dedos (pulgar e índice)
        min_dist: Distancia mínima para apertura completa
        max_dist: Distancia máxima para cierre completo
    """
    # Normalizar y recortar para obtener valor entre 0 (cerrado) y 1 (abierto)
    normalized_distance = np.clip((distance - min_dist) / (max_dist - min_dist), 0.0, 1.0)
    
    # Configurar la señal para el script de Lua en CoppeliaSim
    sim.simxSetFloatSignal(clientID, 'gripper_opening', normalized_distance, sim.simx_opmode_oneshot)

# Calcular distancia euclidiana entre dos puntos
def distance(p1, p2):
    """Calcula distancia euclidiana entre dos puntos"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Función para calcular la profundidad basada en la distancia entre hombro y muñeca
def estimate_depth(shoulder_pos, wrist_pos, min_depth=0.2, max_depth=0.8):
    """
    Estima la profundidad (coordenada z) basada en la distancia entre hombro y muñeca
    Args:
        shoulder_pos: Posición del hombro en coordenadas de la cámara
        wrist_pos: Posición de la muñeca en coordenadas de la cámara
        min_depth: Valor mínimo de profundidad
        max_depth: Valor máximo de profundidad
    Returns:
        Valor estimado de profundidad
    """
    # La idea es que cuando el brazo está estirado (distancia mayor), 
    # el punto está más cerca de la cámara (menor profundidad)
    arm_length = distance(shoulder_pos, wrist_pos)
    
    # Normalizar entre valores esperados de longitud del brazo
    # Estos valores deberías ajustarlos según tus necesidades
    max_arm_length = 300  # pixels
    min_arm_length = 100  # pixels
    
    # Calcular profundidad normalizada e invertida
    norm_length = np.clip((arm_length - min_arm_length) / (max_arm_length - min_arm_length), 0.0, 1.0)
    depth = min_depth + (1.0 - norm_length) * (max_depth - min_depth)
    
    return depth

# Configurar cinemática inversa para el brazo robótico
def setup_inverse_kinematics(clientID, target_handle, tip_handle, base_handle):
    """
    Configura la cinemática inversa para el brazo robótico
    Args:
        clientID: ID de conexión a CoppeliaSim
        target_handle: Handle del objeto target/objetivo
        tip_handle: Handle del efector final
        base_handle: Handle de la base del brazo
    Returns:
        Handle del grupo de IK o None si falla
    """
    try:
        # Crear un nuevo grupo de IK
        result, ikGroupHandle = sim.simxCallScriptFunction(
            clientID, 'remoteApiCommandServer', 
            sim.sim_scripttype_childscript,
            'createIkGroup', [], [], [], bytearray(),
            sim.simx_opmode_blocking)
            
        if result != 0:
            print(f"Error al crear grupo IK: {result}")
            return None
            
        # Crear el elemento de IK (tip -> target)
        result, ikElementHandle = sim.simxCallScriptFunction(
            clientID, 'remoteApiCommandServer', 
            sim.sim_scripttype_childscript,
            'createIkElement', [ikGroupHandle, tip_handle, target_handle, base_handle], 
            [], [], bytearray(),
            sim.simx_opmode_blocking)
            
        if result != 0:
            print(f"Error al crear elemento IK: {result}")
            return None
            
        print(f"Cinemática inversa configurada: Grupo={ikGroupHandle}, Elemento={ikElementHandle}")
        return ikGroupHandle
    except Exception as e:
        print(f"Error al configurar cinemática inversa: {e}")
        return None

# Función para activar el cálculo de cinemática inversa
def solve_inverse_kinematics(clientID, ikGroupHandle):
    """
    Resuelve la cinemática inversa para la posición actual del target
    Args:
        clientID: ID de conexión a CoppeliaSim
        ikGroupHandle: Handle del grupo de IK
    """
    if ikGroupHandle is not None:
        try:
            # Llamar al solucionador de IK
            result = sim.simxCallScriptFunction(
                clientID, 'remoteApiCommandServer', 
                sim.sim_scripttype_childscript,
                'solveIk', [ikGroupHandle], [], [], bytearray(),
                sim.simx_opmode_oneshot)
        except Exception as e:
            print(f"Error al resolver IK: {e}")

# Función para crear un objeto cúbico físico
def create_dynamic_cube(clientID, size=0.05, position=[0,0,0], color=[1,0,0]):
    """
    Crea un cubo con propiedades dinámicas en CoppeliaSim
    Args:
        clientID: ID de conexión a CoppeliaSim
        size: Tamaño del cubo
        position: Posición inicial [x,y,z]
        color: Color [r,g,b] en rango 0-1
    Returns:
        Handle del objeto creado o None si falla
    """
    try:
        # Crear cubo
        emptyBuff = bytearray()
        res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(
            clientID, 'remoteApiCommandServer',
            sim.sim_scripttype_childscript,
            'createCube', [], [size] + position + color, [], emptyBuff,
            sim.simx_opmode_blocking)
            
        if res == 0 and len(retInts) > 0:
            cubeHandle = retInts[0]
            print(f"Cubo creado con éxito: handle={cubeHandle}")
            
            # Configurar propiedades físicas
            cube = PhysicalObject(f"Cube_{cubeHandle}", clientID, cubeHandle)
            cube.enable_dynamics(True)
            
            return cubeHandle
        else:
            print(f"Error al crear cubo: {res}")
            return None
    except Exception as e:
        print(f"Error al crear cubo dinámico: {e}")
        return None

# Función para inicializar y obtener handles de objetos importantes
def initialize_joint_controllers(clientID):
    """
    Inicializa controladores para todos los joints importantes
    Args:
        clientID: ID de conexión a CoppeliaSim
    Returns:
        Diccionario con los controladores de joints
    """
    joint_names = [
        'target',     # Target para el efector final
        'ikTarget',   # Target para cinemática inversa
        'index_finger', # Dedo índice
        'thumb_finger', # Dedo pulgar
        'wrist',      # Muñeca
        'elbow',      # Codo
        'shoulder',   # Hombro
        'arm_base',   # Base del brazo
        'robotBase',  # Base del robot 
        'endEffector' # Efector final
    ]
    
    controllers = {}
    for name in joint_names:
        controller = JointController(name, clientID)
        if controller.get_handle():
            controllers[name] = controller
            print(f"Joint '{name}' inicializado correctamente")
        else:
            print(f"¡ADVERTENCIA! No se pudo obtener handle para '{name}'")
            print(f"Asegúrate de que existe un objeto con ese nombre en tu escena de CoppeliaSim")
    
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

    # Conexión a CoppeliaSim
    clientID = connect_to_coppeliasim()
    if clientID == -1:
        print("No se pudo conectar a CoppeliaSim, saliendo...")
        cap.release()
        sys.exit(1)

    # Inicializar controladores de joints
    try:
        controllers = initialize_joint_controllers(clientID)
        if not controllers:
            print("No se pudieron inicializar los controladores de joints, saliendo...")
            sim.simxFinish(clientID)
            cap.release()
            sys.exit(1)
    except Exception as e:
        print(f"Error al inicializar controladores: {e}")
        sim.simxFinish(clientID)
        cap.release()
        sys.exit(1)
    
    # Configurar cinemática inversa si tenemos los objetos necesarios
    ik_group_handle = None
    if all(k in controllers for k in ['ikTarget', 'endEffector', 'robotBase']):
        ik_group_handle = setup_inverse_kinematics(
            clientID, 
            controllers['ikTarget'].handle, 
            controllers['endEffector'].handle, 
            controllers['robotBase'].handle
        )
    else:
        print("No se pudo configurar la cinemática inversa: faltan objetos necesarios")

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
                    if 'shoulder' in controllers:
                        controllers['shoulder'].update_position(shoulder_pos_3d)
                
                if elbow_pos is not None:
                    elbow_pos_3d = map_coords_to_sim(elbow_pos, estimated_depth, frame_width, frame_height)
                    if last_positions['elbow'] is not None:
                        elbow_pos_3d = [
                            last_positions['elbow'][i] * smoothing_factor + 
                            elbow_pos_3d[i] * (1 - smoothing_factor) for i in range(3)
                        ]
                    last_positions['elbow'] = elbow_pos_3d
                    if 'elbow' in controllers:
                        controllers['elbow'].update_position(elbow_pos_3d)
                
                if wrist_pos is not None:
                    wrist_pos_3d = map_coords_to_sim(wrist_pos, estimated_depth - 0.1, frame_width, frame_height)
                    if last_positions['wrist'] is not None:
                        wrist_pos_3d = [
                            last_positions['wrist'][i] * smoothing_factor + 
                            wrist_pos_3d[i] * (1 - smoothing_factor) for i in range(3)
                        ]
                    last_positions['wrist'] = wrist_pos_3d
                    if 'wrist' in controllers:
                        controllers['wrist'].update_position(wrist_pos_3d)
                
                # Actualizar posición del target de IK
                if wrist_pos is not None and 'ikTarget' in controllers:
                    # El target de IK suele seguir la posición de la muñeca
                    controllers['ikTarget'].update_position(wrist_pos_3d)
                    
                # Resolver IK (esto moverá todo el brazo incluyendo partes no mapeadas directamente)
                if ik_group_handle is not None:
                    solve_inverse_kinematics(clientID, ik_group_handle)

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
                cv2.putText(frame, f"Dist: {finger_distance:.1f}px", 
                          (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
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
                if 'index_finger' in controllers:
                    controllers['index_finger'].update_position(index_pos_3d)
                
                if 'thumb_finger' in controllers:
                    controllers['thumb_finger'].update_position(thumb_pos_3d)
                
                # Actualizar posición del target (punto medio entre índice y pulgar)
                if 'target' in controllers:
                    target_pos = [(index_pos_3d[i] + thumb_pos_3d[i])/2 for i in range(3)]
                    controllers['target'].update_position(target_pos)
                
                # Actualizar estado del gripper basado en distancia entre dedos
                update_gripper_state(clientID, finger_distance)
                
                # Guardar posiciones para suavizado
                last_positions['index'] = index_pos_3d
                last_positions['thumb'] = thumb_pos_3d