
<p style="text-align: justify;">Canny_EdgeVideo.py carga un archivo de vídeo y aplica detección de bordes de Canny junto con un filtro de Kalman para resaltar y seguir los centroides de los contornos. Al definir una tasa de fotogramas objetivo, salta cuadros según sea necesario, convierte cada imagen procesada a escala de grises y calcula los bordes para generar una vista clara y filtrada de las formas en movimiento.</p>

<p style="text-align: justify;">Para cada fotograma, se extraen los contornos del mapa de bordes y se calculan sus centroides mediante momentos de imagen. El primer centroide inicializa el estado del filtro de Kalman, y cada centroide posterior se utiliza como medición para predecir y corregir la estimación del filtro. Este paso de suavizado reduce las oscilaciones en las posiciones rastreadas a lo largo del tiempo.</p>

<p style="text-align: justify;">La visualización superpone el mapa de bordes sobre el fotograma en escala de grises, dibujando cada contorno y marcando su centroide en blanco.</p>

<p style="text-align: justify;">El código y un vídeo de ejemplo se encuentran en la carpeta <code>Canny2</code>. Para ejecutar el ejemplo, entrar en esa carpeta y ejecutar:<br><pre><code>bash  
python3 Canny_EdgeVideo.py  
</code></pre>Los parámetros al inicio del script permiten ajustar los umbrales de Canny, los FPS objetivo y el retardo entre cuadros para adaptarse al vídeo y necesidades de rendimiento.</p>


Para mi versión tengo dos codigo principales y también empecé a entrenar un modelo de YOLO en Roboflow, pero le di más frames por segundo así que no va a estar lsito a tiempo. Lo que si está por ahora es el uso de Canny Edges y HoughLines para ambos codigos que subí (contienen dentro comentads varias versiones de si mismos). Usamos el algoritmo de Canny para detectar bordes una ves generamos un enfoque Gaussiano en un frame del video en gris. Si no entiendo mal el video esta a 24 frames per second en la versión final. Lo que causa una buena carga. Pero a cambio detecta una buena parte del tiempo el color exacto que buscamos de las patitos. Lamentablemente sin la implementación de Yolo no puede detectar en si que es un "patito" y confunde su color con el de la sombra de la banca y unos pantalones. Por eso buscamos usar Yolo con un modelo ya entrenado e importado en .yaml.
