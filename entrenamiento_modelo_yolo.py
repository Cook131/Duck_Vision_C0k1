from ultralytics import YOLO
import os
import yaml
from pathlib import Path
import shutil

# --- Parámetros de entrenamiento ---
# IMPORTANTE: Asegúrate de que esta ruta apunte a tu archivo YAML
DATASET_YAML = '/Users/jorgenajera/Documents/Duck_vision/dataset/patitos.yaml'  # Ruta al archivo YAML
EPOCHS = 50                          # Número de épocas de entrenamiento
BATCH_SIZE = 16                      # Tamaño del lote
IMG_SIZE = 640                       # Tamaño de imagen para entrenamiento
DEVICE = 'cpu'                       # Usar CPU en MacBook, cambia a 0 si tienes GPU
PROJECT = 'patitos_detector'         # Nombre del proyecto
NAME = 'yolov8n_patitos'             # Nombre del experimento

# Verificación de estructura de datos adaptada para formato Roboflow
def verify_dataset(yaml_path):
    """Verifica que el dataset esté correctamente estructurado"""
    print(f"Verificando dataset en: {yaml_path}")
    
    # Comprobar si el archivo YAML existe
    if not os.path.isfile(yaml_path):
        print(f"Error: El archivo YAML no existe: {yaml_path}")
        print(f"Directorio actual: {os.getcwd()}")
        print(f"Contenido del directorio: {os.listdir(os.path.dirname(yaml_path) if os.path.dirname(yaml_path) else '.')}")
        return False
        
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            print(f"Contenido del YAML: {data}")
            
        # Verificar campos obligatorios para formato Roboflow
        required = ['train', 'val', 'names']
        if not all(field in data for field in required):
            print(f"Error: El archivo YAML debe contener los campos: {required}")
            return False
            
        # Obtener directorio base del archivo YAML
        yaml_dir = Path(yaml_path).parent
        print(f"Directorio base del YAML: {yaml_dir}")
        
        # Construir rutas absolutas desde las rutas relativas en el YAML
        train_rel_path = data['train']
        val_rel_path = data['val']
        
        # Eliminar '../' si está presente
        if train_rel_path.startswith('../'):
            train_rel_path = train_rel_path[3:]
        if val_rel_path.startswith('../'):
            val_rel_path = val_rel_path[3:]
            
        # Construir rutas absolutas
        train_path = yaml_dir.parent / train_rel_path
        val_path = yaml_dir.parent / val_rel_path
        
        print(f"Verificando directorio de entrenamiento: {train_path}")
        if not train_path.exists():
            print(f"Error: Directorio de entrenamiento no encontrado: {train_path}")
            # Intentar buscar en ubicaciones alternativas
            alt_train_path = yaml_dir / train_rel_path
            print(f"Intentando ubicación alternativa: {alt_train_path}")
            if alt_train_path.exists():
                train_path = alt_train_path
                print("Ubicación alternativa encontrada correctamente.")
            else:
                print("No se pudo encontrar el directorio de entrenamiento.")
                return False
            
        print(f"Verificando directorio de validación: {val_path}")
        if not val_path.exists():
            print(f"Error: Directorio de validación no encontrado: {val_path}")
            # Intentar ubicación alternativa
            alt_val_path = yaml_dir / val_rel_path
            print(f"Intentando ubicación alternativa: {alt_val_path}")
            if alt_val_path.exists():
                val_path = alt_val_path
                print("Ubicación alternativa encontrada correctamente.")
            else:
                print("No se pudo encontrar el directorio de validación.")
                return False
            
        print("Estructura del dataset verificada correctamente.")
        print(f"Clases: {data['names']}")
        print(f"Imágenes de entrenamiento: {train_path}")
        print(f"Imágenes de validación: {val_path}")

        # Contar imágenes y etiquetas
        train_imgs = list(train_path.glob('*.jpg')) + list(train_path.glob('*.png'))
        train_labels = list(train_path.glob('*.txt'))
        val_imgs = list(val_path.glob('*.jpg')) + list(val_path.glob('*.png'))
        val_labels = list(val_path.glob('*.txt'))
        
        print(f"Entrenamiento: {len(train_imgs)} imágenes, {len(train_labels)} etiquetas")
        print(f"Validación: {len(val_imgs)} imágenes, {len(val_labels)} etiquetas")
        
        # Agregar el campo 'path' si no existe
        if 'path' not in data:
            data['path'] = str(yaml_dir.parent)
            print(f"Agregando campo 'path' al YAML: {data['path']}")
            
            # Actualizar el archivo YAML
            with open(yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
                print("Archivo YAML actualizado con campo 'path'.")
        
        return True
    except Exception as e:
        print(f"Error al verificar el dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_yolov8(yaml_path):
    """Entrena un modelo YOLOv8 con el dataset proporcionado"""
    # Cargar modelo base (pre-entrenado)
    model = YOLO('yolov8n.pt')  # puedes cambiar a 's', 'm', 'l' o 'x' para modelos más grandes
    
    # Configurar y comenzar entrenamiento
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        verbose=True,
        patience=15,          # Detención temprana si no hay mejora en 15 épocas
        save=True,            # Guardar checkpoints
        save_period=10,       # Guardar cada 10 épocas
        pretrained=True,      # Usar pesos preentrenados
        optimizer='Adam',     # Optimizador (SGD, Adam, AdamW)
        lr0=0.001,            # Tasa de aprendizaje inicial
        lrf=0.01,             # Tasa de aprendizaje final (como fracción de lr0)
        momentum=0.937,       # Momentum para SGD
        weight_decay=0.0005,  # Weight decay
        warmup_epochs=3,      # Épocas de calentamiento
        cos_lr=True,          # Usar programación de tasa de aprendizaje coseno
        augment=True,         # Usar aumentación de datos
        cache=False,          # No cachear imágenes (para datasets más grandes)
        rect=False,           # No usar entrenamiento rectangular
        resume=False,         # No reanudar desde último checkpoint
        amp=True,             # Precision mixta automática
        fraction=1.0,         # Usar todo el dataset
        profile=False,        # No perfilar rendimiento
        overlap_mask=True,    # Máscaras solapadas para segmentación
        mask_ratio=4,         # Ratio de downsampling para máscaras
        dropout=0.0,          # Sin dropout
        val=True,             # Validar durante entrenamiento
        plots=True            # Generar gráficos de entrenamiento
    )
    
    return results

def export_model(results_dir):
    """Exporta el modelo entrenado a diferentes formatos"""
    # Buscar el mejor modelo (mejor.pt)
    best_model_path = None
    for path in Path(results_dir).rglob('*.pt'):
        if 'best' in path.name:
            best_model_path = path
            break
    
    if best_model_path is None:
        print("No se encontró el mejor modelo para exportar.")
        return False
    
    # Cargar y exportar modelo
    print(f"Exportando modelo desde: {best_model_path}")
    model = YOLO(best_model_path)
    
    # Exportar a diferentes formatos
    formats = ['torchscript', 'onnx']  # Reducido para Mac, añade 'openvino' si necesitas
    for format_type in formats:
        try:
            model.export(format=format_type)
            print(f"Modelo exportado a formato {format_type}")
        except Exception as e:
            print(f"Error al exportar a {format_type}: {e}")
    
    return True

# --- Función principal ---
def main():
    print("=== Entrenamiento de YOLOv8 para detección de patitos ===")
    
    # Verificar dataset
    if not verify_dataset(DATASET_YAML):
        print("Corrige los problemas en el dataset antes de continuar.")
        return
    
    # Preguntar al usuario si desea continuar
    respuesta = input("\n¿Deseas iniciar el entrenamiento? (s/n): ")
    if respuesta.lower() != 's':
        print("Entrenamiento cancelado.")
        return
    
    # Entrenar modelo
    print("\nIniciando entrenamiento...")
    results = train_yolov8(DATASET_YAML)
    
    # Exportar modelo
    print("\nEntrenamiento completado. Exportando modelo...")
    export_success = export_model(f"{PROJECT}/{NAME}")
    
    if export_success:
        print("\n=== Proceso completado con éxito ===")
        print(f"El modelo entrenado se encuentra en: {PROJECT}/{NAME}")
        print("Puedes usar este modelo en el script de detección.")
    else:
        print("\n=== Proceso completado con advertencias ===")
        print("Revisa los mensajes anteriores para identificar posibles problemas.")

if __name__ == "__main__":
    main()