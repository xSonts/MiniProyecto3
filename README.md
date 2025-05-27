# 🍱 Clasificador de Comida con TensorFlow Lite y Raspberry Pi

Este proyecto permite identificar diferentes tipos de platos de comida en tiempo real usando una cámara, un modelo de aprendizaje profundo en formato TensorFlow Lite y una interfaz gráfica desarrollada con Tkinter. 

> 📷 Toma una foto con la webcam, clasifica el plato de comida y muestra su nombre, ingredientes y posibles alérgenos.

---

## 🧠 Modelo

Se utilizó una red neuronal basada en **MobileNetV2** entrenada para clasificar 5 tipos de comidas:
- 🥗 Caesar Salad  
- 🐟 Ceviche  
- 🧆 Hummus  
- 🍨 Ice Cream  
- 🍣 Sushi  

El modelo fue convertido a **TensorFlow Lite** para una ejecución eficiente en dispositivos como Raspberry Pi.

---

## 🛠️ Requisitos

### Software
- Python 3.8+
- TensorFlow Lite
- OpenCV (`cv2`)
- Pillow (`PIL`)
- Tkinter (incluido con Python)
  
Instalación de dependencias:
```bash
pip install opencv-python pillow

├── datos_platos.py         # Diccionario con información de cada plato
├── modelo_final.tflite     # Modelo TFLite entrenado para la clasificación
├── app.py                  # Aplicación principal con la interfaz gráfica
├── Imgs/                   # Carpeta con imágenes de referencia de cada plato
└── README.md

