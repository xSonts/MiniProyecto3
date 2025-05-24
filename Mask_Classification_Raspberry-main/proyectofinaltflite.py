import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk
from datos_platos import platos_info

class_names = list(platos_info.keys())

interpreter = tf.lite.Interpreter(model_path=r"E:\Universidad\IA\Mask_Classification_Raspberry-main\Mask_Classification_Raspberry-main\modelo_final.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
target_height = input_shape[1]
target_width = input_shape[2]

class FoodClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🍱 Clasificador de Comida")
        self.root.configure(bg="#e8f6ff")  # Azul muy claro

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Layout principal
        self.video_label = tk.Label(root, bg="#d0eaff", bd=2, relief="solid")
        self.video_label.grid(row=0, column=0, padx=20, pady=20)

        self.capture_btn = tk.Button(root, text="📸 Capturar (Espacio)", command=self.capture_image,
                                     font=("Arial", 11, "bold"), bg="#4caf50", fg="white", padx=10, pady=5, relief="raised")
        self.capture_btn.grid(row=1, column=0, pady=(0, 20))

        # Info del plato
        self.info_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove")
        self.info_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=20, sticky="n")

        self.image_label = tk.Label(self.info_frame, bg="#ffffff")
        self.image_label.pack(pady=10)

        self.info_label = tk.Label(self.info_frame, text="", font=("Arial", 12, "bold"), bg="#ffffff",
                                   wraplength=300, justify="left", anchor="n", fg="#333")
        self.info_label.pack(padx=10, pady=10)

        # Evento para tecla espacio
        self.root.bind("<space>", lambda event: self.capture_image())

        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (320, 240))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_video)

    def capture_image(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        img_resized = cv2.resize(frame, (target_width, target_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(img_normalized, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = class_names[np.argmax(output_data)]

        self.mostrar_info_plato(prediction)

    def mostrar_info_plato(self, nombre_plato):
        datos = platos_info[nombre_plato]
        nombre_formateado = nombre_plato.replace('_', ' ').title()
        descripcion = (
            f"🍽️ {nombre_formateado}\n\n"
            f"{datos['descripcion']}\n\n"
            f"🍅 Ingredientes:\n• " + "\n• ".join(datos['ingredientes']) + "\n\n"
            f"⚠️ Alergenos:\n• " + "\n• ".join(datos['alergenos'])
        )
        self.info_label.config(text=descripcion)

        img = Image.open(datos['imagen']).resize((224, 224))
        imgtk = ImageTk.PhotoImage(img)
        self.image_label.imgtk = imgtk
        self.image_label.configure(image=imgtk)

# Ejecutar app
root = tk.Tk()
app = FoodClassifierApp(root)
root.mainloop()
