# Canny Edge

<code>Canny2/Canny_EdgeVideo.py</code> carga un vídeo y aplica detección de bordes de **Canny** junto a un filtro de **Kalman** para resaltar y seguir los centroides de los contornos.

## ¿Cómo funciona?

1. **Extracción de contornos**  
   Cada fotograma se convierte a escala de grises y se aplica Canny para obtener un mapa de bordes.  
2. **Cálculo de centroides**  
   A partir de los momentos de imagen, se determina el centroide de cada contorno detectado.  
3. **Filtro de Kalman**  
   - El primer centroide inicializa el estado del filtro.  
   - Cada medición posterior (nuevo centroide) se utiliza para predecir y corregir la posición estimada, suavizando las oscilaciones.  
4. **Visualización**  
   Se superpone el mapa de bordes sobre el fotograma en gris, dibujando cada contorno y marcando su centroide con un punto blanco.

Coloca un vídeo de ejemplo en la carpeta <code>Canny2/video.mp4</code>, navega hasta ella y ejecuta:

```bash
python3 Canny_EdgeVideo.py
```

![Canny Edge Detection](https://github.com/user-attachments/assets/2443e45b-8fff-40ce-a46f-21987da4b759)
