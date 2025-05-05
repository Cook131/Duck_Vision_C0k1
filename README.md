## Descripción

*Canny_EdgeVideo.py* carga un vídeo y aplica detección de bordes de **Canny** junto a un filtro de **Kalman** para resaltar y seguir los centroides de los contornos. Gracias al parámetro de FPS objetivo, el script omite fotogramas cuando es necesario, convierte cada imagen a escala de grises y detecta bordes, ofreciendo así una vista nítida de las formas en movimiento.

## ¿Cómo funciona?

1. **Extracción de contornos**  
   Cada fotograma se convierte a escala de grises y se aplica Canny para obtener un mapa de bordes.  
2. **Cálculo de centroides**  
   A partir de los momentos de imagen, se determina el centroide de cada contorno detectado.  
3. **Filtro de Kalman**  
   - El primer centroide inicializa el estado del filtro.  
   - Cada medición posterior (nuevo centroide) se utiliza para predecir y corregir la posición estimada, suavizando las oscilaciones.  
4. **Visualización**  
   Se superpone el mapa de bordes sobre el fotograma en gris, dibujando cada contorno y marcando su centroide con un punto blanco. Para salir, basta con pulsar `q`.

## Uso

Coloca el código y un vídeo de ejemplo en la carpeta `Canny2`, navega hasta ella y ejecuta:

```bash
python3 Canny_EdgeVideo.py
