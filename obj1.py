import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection System")
        self.root.geometry("1000x700")
        
        # Initialize YOLO model
        self.model = YOLO('yolov8s.pt')  # Will auto-download if not found
        
        # UI Elements
        self.create_widgets()
        
        # Video capture variables
        self.cap = None
        self.is_running = False
        self.current_mode = None  # 'image', 'video', or 'camera'
        
        # Detection colors
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128)
        ]
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Control frame
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(fill=tk.X)
        
        # Mode selection
        tk.Label(control_frame, text="Select Mode:").grid(row=0, column=0, padx=5)
        
        self.mode_var = tk.StringVar(value="image")
        tk.Radiobutton(control_frame, text="Image", variable=self.mode_var, value="image", 
                      command=self.mode_changed).grid(row=0, column=1, padx=5)
        tk.Radiobutton(control_frame, text="Video File", variable=self.mode_var, value="video", 
                      command=self.mode_changed).grid(row=0, column=2, padx=5)
        tk.Radiobutton(control_frame, text="Live Camera", variable=self.mode_var, value="camera", 
                      command=self.mode_changed).grid(row=0, column=3, padx=5)
        
        # File selection
        self.file_button = tk.Button(control_frame, text="Select File", command=self.select_file)
        self.file_button.grid(row=0, column=4, padx=5)
        
        # Start/Stop buttons
        self.start_button = tk.Button(control_frame, text="Start Detection", command=self.start_detection)
        self.start_button.grid(row=0, column=5, padx=5)
        self.stop_button = tk.Button(control_frame, text="Stop", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=6, padx=5)
        
        # Display area
        self.display_frame = tk.Frame(self.root, bg='black')
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.display_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Detection info
        info_frame = tk.Frame(self.root)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(info_frame, text="Detected Objects:").pack(side=tk.LEFT)
        self.detection_label = tk.Label(info_frame, text="None", fg="blue")
        self.detection_label.pack(side=tk.LEFT, padx=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X)
        
        # Set initial mode
        self.mode_changed()
    
    def mode_changed(self):
        """Handle mode selection change"""
        self.current_mode = self.mode_var.get()
        
        if self.current_mode == "camera":
            self.file_button.config(state=tk.DISABLED)
            self.start_button.config(state=tk.NORMAL)
        else:
            self.file_button.config(state=tk.NORMAL)
            self.start_button.config(state=tk.DISABLED)
        
        self.status_var.set(f"Mode: {self.current_mode.capitalize()}. {'Select a file' if self.current_mode != 'camera' else 'Ready to start'}")
    
    def select_file(self):
        """Select image or video file based on current mode"""
        filetypes = []
        if self.current_mode == "image":
            filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp")]
        elif self.current_mode == "video":
            filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv")]
        
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.file_path = filename
            self.start_button.config(state=tk.NORMAL)
            self.status_var.set(f"Selected: {os.path.basename(filename)}")
            
            # Preview for images
            if self.current_mode == "image":
                img = Image.open(filename)
                img.thumbnail((800, 600))
                self.preview_img = ImageTk.PhotoImage(img)
                self.canvas.create_image(400, 300, image=self.preview_img, anchor=tk.CENTER)
    
    def start_detection(self):
        """Start detection based on current mode"""
        if self.current_mode == "camera":
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
        elif self.current_mode == "video":
            self.cap = cv2.VideoCapture(self.file_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video file")
                return
        elif self.current_mode == "image":
            self.process_image(self.file_path)
            return
        
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.file_button.config(state=tk.DISABLED)
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detect_objects)
        self.detection_thread.daemon = True
        self.detection_thread.start()
    
    def stop_detection(self):
        """Stop ongoing detection"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.file_button.config(state=tk.NORMAL)
        self.status_var.set("Detection stopped")
    
    def process_image(self, image_path):
        """Process a single image"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                messagebox.showerror("Error", "Could not read image file")
                return
            
            # Run detection
            results = self.model(img)
            
            # Draw detections
            detected_objects = set()
            for result in results:
                for box in result.boxes:
                    if box.conf[0] > 0.5:  # Confidence threshold
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        label = result.names[cls]
                        color = self.colors[cls % len(self.colors)]
                        
                        # Draw rectangle
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        cv2.putText(img, f"{label} {float(box.conf[0]):.2f}", 
                                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, color, 2)
                        
                        detected_objects.add(label)
            
            # Convert to RGB for display
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img.thumbnail((800, 600))
            self.result_img = ImageTk.PhotoImage(img)
            
            # Update display
            self.canvas.delete("all")
            self.canvas.create_image(400, 300, image=self.result_img, anchor=tk.CENTER)
            
            # Update detection info
            if detected_objects:
                self.detection_label.config(text=", ".join(detected_objects))
            else:
                self.detection_label.config(text="No objects detected")
            
            self.status_var.set("Image processed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def detect_objects(self):
        """Main detection loop for video and camera"""
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                if self.current_mode == "video":
                    messagebox.showinfo("Info", "Video processing completed")
                break
            
            # Run detection
            results = self.model.track(frame, persist=True)
            
            # Draw detections
            detected_objects = set()
            for result in results:
                for box in result.boxes:
                    if box.conf[0] > 0.5:  # Confidence threshold
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        label = result.names[cls]
                        color = self.colors[cls % len(self.colors)]
                        
                        # Draw rectangle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        cv2.putText(frame, f"{label} {float(box.conf[0]):.2f}", 
                                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, color, 2)
                        
                        detected_objects.add(label)
            
            # Convert to RGB for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img.thumbnail((800, 600))
            self.video_img = ImageTk.PhotoImage(img)
            
            # Update display in main thread
            self.root.after(0, self.update_display, self.video_img, detected_objects)
            
            # Control frame rate
            cv2.waitKey(1)
        
        self.stop_detection()
    
    def update_display(self, img, detected_objects):
        """Update the display with new frame and detection info"""
        self.canvas.delete("all")
        self.canvas.create_image(400, 300, image=img, anchor=tk.CENTER)
        
        if detected_objects:
            self.detection_label.config(text=", ".join(detected_objects))
        else:
            self.detection_label.config(text="No objects detected")
        
        self.status_var.set(f"Detecting in {self.current_mode} mode...")

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
