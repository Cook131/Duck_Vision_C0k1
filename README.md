# Visión Computacional - Implementación de robótica inteligente

Retos de visión computacional desarrollados para el curso "Implementación de robótica inteligente."

## Canny Edge Detection

<code>Canny2/Canny_EdgeVideo.py</code> carga un vídeo y aplica detección de bordes de **Canny** junto a un filtro de **Kalman** para resaltar y seguir los centroides de los contornos.

### ¿Cómo funciona?

1. **Extracción de contornos**  
   Cada fotograma se convierte a escala de grises y se aplica Canny para obtener un mapa de bordes.  
2. **Cálculo de centroides**  
   A partir de los momentos de imagen, se determina el centroide de cada contorno detectado.  
3. **Filtro de Kalman**  
   - El primer centroide inicializa el estado del filtro.  
   - Cada medición posterior (nuevo centroide) se utiliza para predecir y corregir la posición estimada, suavizando las oscilaciones.  
4. **Visualización**  
   Se superpone el mapa de bordes sobre el fotograma en gris, dibujando cada contorno y marcando su centroide con un punto blanco.

### Uso

Coloca un vídeo de ejemplo <code>video.mp4</code> en la carpeta <code>Canny2</code>, navega hasta ella y ejecuta:

```bash
python3 Canny_EdgeVideo.py
```
<p align="center">
  <img src="https://github.com/user-attachments/assets/2443e45b-8fff-40ce-a46f-21987da4b759" alt="Canny Edge Detection Demo" width="600"/>
</p>




# 🤖 Control de NiryoOne en CoppeliaSim con Python, MediaPipe y ZeroMQ API

Este proyecto permite controlar el brazo robótico NiryoOne en CoppeliaSim usando los movimientos de tu brazo capturados por webcam y MediaPipe. Además, puedes controlar la pinza para agarrar objetos virtuales. Todo se comunica usando la moderna ZeroMQ Remote API.

## 🛠️ Técnicas y Tecnologías Usadas

- **MediaPipe (Pose & Hands)**:  
  Para detectar y rastrear en tiempo real los puntos clave del brazo y la mano usando la webcam.
- **OpenCV**:  
  Para la captura y visualización de video.
- **ZeroMQ Remote API**:  
  Comunicación moderna y eficiente entre Python y CoppeliaSim.
- **CoppeliaSim (V-REP)**:  
  Simulación física y visual del robot NiryoOne.
- **Lua Scripting**:  
  Control directo de la pinza (gripper) dentro de la simulación.
- **Kinematics Addon**:  
  Para la cinemática inversa (IK) del robot, ya que la versión usada de CoppeliaSim no tiene soporte nativo de IK Groups.

## ⚡ Implementación del ZeroMQ Remote API

- Se eliminó todo el código legado (`simx*`) y se migró completamente a la API ZeroMQ:
  ```python
  from coppeliasim_zmqremoteapi_client import RemoteAPIClient
  client = RemoteAPIClient()
  sim = client.require('sim')
  ```
- Todos los objetos y señales se manejan usando los métodos modernos de la API (`getObject`, `setObjectPosition`, `setFloatSignal`, etc).

## 🏗️ Cambios y Configuración en CoppeliaSim

### 1. **Renombrar Objetos y Estructura**
- Se aseguraron los siguientes nombres (¡importante!):
  - `/NiryoOne/NiryoOne_target` (dummy para el target IK)
  - `/NiryoOne/NiryoOne_tip` (dummy para el tip IK)
  - `/NiryoOne/NiryoOne_joint1` ... `/NiryoOne/NiryoOne_joint6` (joints del robot)
  - `/NiryoOne/NiryoOneGripper` (pinza)
  - `/NiryoOne/NiryoOneGripper/leftJoint1` y `/rightJoint1` (joints de los dedos)

### 2. **Configuración de Cinemática Inversa (IK)**
- Se crearon los dummies `tip` y `target` y se colocaron correctamente en la jerarquía del robot.
- Se usó el **Kinematics Addon** para crear y configurar el grupo IK, ya que la versión de CoppeliaSim no tiene soporte nativo.
- El script Python crea y resuelve el grupo IK en cada ciclo para mover el brazo según la posición de la muñeca detectada.

### 3. **Script Lua para la Pinza 🦾**
- Se añadió un script Lua como child script en el objeto `NiryoOneGripper`:
  ```lua
  function sysCall_init()
      leftFinger = sim.getObject('../leftJoint1')
      rightFinger = sim.getObject('../rightJoint1')
      minPos = 0
      maxPos = 0.02
  end

  function sysCall_actuation()
      local opening = sim.getFloatSignal('gripper_opening')
      if opening ~= nil then
          local pos = minPos + (maxPos - minPos) * opening
          sim.setJointTargetPosition(leftFinger, pos)
          sim.setJointTargetPosition(rightFinger, pos)
      end
  end
  ```
- El script recibe la señal `gripper_opening` desde Python y ajusta la apertura de la pinza en tiempo real.

## 🧠 Lógica de Control (Python)

- **MediaPipe** detecta hombro, codo, muñeca, pulgar e índice.
- Se mapean las coordenadas 2D de la cámara a 3D en el espacio de CoppeliaSim.
- Solo la posición de la muñeca (wrist) actualiza el target IK para movimientos más naturales.
- La distancia entre pulgar e índice controla la apertura de la pinza.

## 📝 Notas y Consejos

- Ajusta los factores de escala y profundidad en el script Python para que el movimiento sea más natural según tu cámara y escena.
- Asegúrate de que los nombres de los objetos en CoppeliaSim coincidan exactamente con los usados en el script.
- Si el robot no se mueve correctamente, revisa la configuración del grupo IK y la jerarquía de los dummies.

## 🚀 ¡Listo para usar!

1. Abre CoppeliaSim y carga la escena con el NiryoOne y los objetos renombrados.
2. Ejecuta el script Python.


# 🏺 Reconstrucción 3D de Figurilla Maya a partir de Video

Este proyecto permite reconstruir una nube de puntos 3D de una figurilla maya usando técnicas de Structure from Motion (SfM) a partir de un video que recorre varios ángulos del objeto.

---

## 🚀 Tecnologías utilizadas

- **Python 3**
- **OpenCV** – Detección de características, extracción de frames, visualización de matches
- **Open3D** – Visualización y manejo de nubes de puntos
- **NumPy** – Procesamiento numérico
- **Matplotlib** – Gráficas comparativas de parámetros
- **scikit-learn** – Ajuste de parámetros y optimización
- **SciPy** – Optimización de nubes de puntos

---

## 📂 Descripción de scripts

- **`sacar_fps.py`**
  - Extrae frames del video original, asegurando buena cobertura angular y calidad de imagen.
- **`deteccion_limites.py`**
  - Contiene la clase principal `SfMReconstructor` que realiza:
    - Detección de características (SIFT/ORB)
    - Emparejamiento de puntos clave entre imágenes
    - Pruebas automáticas de parámetros RANSAC
    - Reconstrucción 3D inicial y adición de vistas
    - Bundle adjustment y exportación a `.ply`
- **`implementación_bici.py`**
  - Script principal para ejecutar todo el pipeline de reconstrucción 3D.
- **`ver_nube.py`**
  - Visualiza la nube de puntos 3D generada (`.ply`) usando Open3D.
  - Permite limpiar, downsamplear y comparar nubes inicial/final.

---

## 🖼️ Ejemplo de resultados

### Nube de puntos 3D reconstruida

<p align="center">
  <!-- Inserta aquí tu gif de la nube de puntos -->
  <img src="[ruta/a/tu_gif_nube.gif](https://github.com/user-attachments/assets/d89afcfa-6e5d-460a-9fec-ee6ef79ff168)" width="500"/>
</p>

### Ejemplo de matches y keypoints entre imágenes

<p align="center">
  <!-- Inserta aquí tus imágenes de matches -->
  <img src="https://github.com/user-attachments/assets/5a01a0d6-894d-4eef-8989-6820f376793d" width="300"/>
  <img src="https://github.com/user-attachments/assets/7575708c-629f-4130-a5c4-6bc19d6af6a7" width="300"/>
</p>

---

## 📝 Pasos de uso

1. **Extrae los frames del video**
   ```bash
   python sacar_fps.py
   ```
2. **Ejecuta la reconstrucción 3D**
   ```bash
   python implementación_bici.py
   ```
3. **Visualiza la nube de puntos**
   ```bash
   python ver_nube.py
   ```

---

## 📌 Notas y recomendaciones

- Usa videos con buena iluminación y fondo neutro para mejores resultados.
- Puedes ajustar los parámetros de detección y matching en `deteccion_limites.py`.
- Si la nube de puntos es pobre, prueba recortar las imágenes o usar menos frames.

---
