# Importing Libraries
import numpy as np
import math
import cv2
import os
import traceback
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import tkinter as tk
from PIL import Image, ImageTk

# Set up Hand Detector
hd = HandDetector(maxHands=1)

# Application class
class Application:
    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.model = load_model('cnn8grps_rad1_model.h5')  # Update the model path as needed
        self.ct = {char: 0 for char in ascii_uppercase}
        self.ct['blank'] = 0
        self.str = " "
        self.current_symbol = "C"

        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.geometry("1300x700")

        # Create GUI elements
        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=3, width=480, height=640)

        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=700, y=115, width=400, height=400)

        self.T = tk.Label(self.root, text="Sign Language To Text Conversion", font=("Courier", 30, "bold"))
        self.T.place(x=60, y=5)

        self.panel3 = tk.Label(self.root)  # Current Symbol
        self.panel3.place(x=280, y=585)

        self.T1 = tk.Label(self.root, text="Character :", font=("Courier", 30, "bold"))
        self.T1.place(x=10, y=580)

        self.panel5 = tk.Label(self.root)  # Sentence
        self.panel5.place(x=260, y=632)

        self.T3 = tk.Label(self.root, text="Sentence :", font=("Courier", 30, "bold"))
        self.T3.place(x=10, y=632)

        self.clear = tk.Button(self.root, text="Clear", font=("Courier", 20), command=self.clear_fun)
        self.clear.place(x=1205, y=630)

        self.video_loop()

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if ok:
                cv2image = cv2.flip(frame, 1)
                hands, frame = hd.findHands(cv2image, draw=False)

                if hands:
                    hand = hands[0]  # Get the first detected hand
                    x, y, w, h = hand['bbox']  # Get bounding box of the hand

                    # Crop the hand region
                    roi = cv2image[y:y + h, x:x + w]
                    roi = cv2.resize(roi, (400, 400))  # Resize to match model input size

                    # Make predictions
                    self.predict(roi)

                    # Draw bounding box around hand
                    cv2.rectangle(cv2image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Convert BGR to RGBA
                cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
                self.current_image = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=self.current_image)
                self.panel.imgtk = imgtk
                self.panel.config(image=imgtk)

                # Update symbol display
                self.panel3.config(text=self.current_symbol, font=("Courier", 30))

        except Exception:
            print("==", traceback.format_exc())
        finally:
            self.root.after(1, self.video_loop)

    def predict(self, test_image):
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension
        test_image = test_image / 255.0  # Normalize

        # Make prediction
        prob = self.model.predict(test_image)[0]
        ch1 = np.argmax(prob)  # Get the index of the highest probability
        self.current_symbol = chr(ch1 + 65)  # Convert index to corresponding character (A=65)

    def clear_fun(self):
        self.str = " "
        self.current_symbol = "C"

    def destructor(self):
        self.vs.release()
        cv2.destroyAllWindows()
        self.root.destroy()

    if __name__ == "__main__":
        app = Application()
        app.root.mainloop()
        app.destructor()