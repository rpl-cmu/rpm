# Claude wrote all of this
# In a single try

import os
import json
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import math

class OrientedBBoxTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Oriented Bounding Box Tool")
        
        # Initialize variables
        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.image_scale = 1.0
        self.boxes = []  # List of oriented bounding boxes
        self.points = []  # Temporary points for current box
        self.is_drawing = False
        self.config_file = None
        
        # Create GUI components
        self.create_menu()
        self.create_canvas()
        self.create_toolbar()
        
        # Bind events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.root.bind("<Control-s>", lambda e: self.save_config())
        self.root.bind("<Control-o>", lambda e: self.load_config())
        
        # Status message
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Open an image to start.")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def create_menu(self):
        """Create the top menu bar"""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Config", command=self.save_config, accelerator="Ctrl+S")
        file_menu.add_command(label="Load Config", command=self.load_config, accelerator="Ctrl+L")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Clear All Boxes", command=self.clear_all_boxes)
        edit_menu.add_command(label="Delete Last Box", command=self.delete_last_box)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Zoom In", command=self.zoom_in)
        view_menu.add_command(label="Zoom Out", command=self.zoom_out)
        view_menu.add_command(label="Reset Zoom", command=self.reset_zoom)
        menubar.add_cascade(label="View", menu=view_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Instructions", command=self.show_instructions)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)

    def create_canvas(self):
        """Create the main canvas for image display"""
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollbars
        self.h_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.v_scrollbar = tk.Scrollbar(self.canvas_frame)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create canvas
        self.canvas = tk.Canvas(
            self.canvas_frame, 
            bg="gray", 
            xscrollcommand=self.h_scrollbar.set,
            yscrollcommand=self.v_scrollbar.set
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Configure scrollbars
        self.h_scrollbar.config(command=self.canvas.xview)
        self.v_scrollbar.config(command=self.canvas.yview)

    def create_toolbar(self):
        """Create a toolbar with controls"""
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        # Box color picker
        tk.Label(toolbar, text="Box Color:").pack(side=tk.LEFT, padx=5, pady=5)
        self.color_var = tk.StringVar(value="red")
        colors = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange", "purple"]
        color_menu = tk.OptionMenu(toolbar, self.color_var, *colors)
        color_menu.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Line thickness
        tk.Label(toolbar, text="Line Thickness:").pack(side=tk.LEFT, padx=5, pady=5)
        self.thickness_var = tk.IntVar(value=2)
        thickness_menu = tk.OptionMenu(toolbar, self.thickness_var, 1, 2, 3, 4, 5)
        thickness_menu.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Buttons
        open_button = tk.Button(toolbar, text="Open Image", command=self.open_image)
        open_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        save_button = tk.Button(toolbar, text="Save Config", command=self.save_config)
        save_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        load_button = tk.Button(toolbar, text="Load Config", command=self.load_config)
        load_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        clear_button = tk.Button(toolbar, text="Clear All", command=self.clear_all_boxes)
        clear_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        delete_button = tk.Button(toolbar, text="Delete Last", command=self.delete_last_box)
        delete_button.pack(side=tk.LEFT, padx=5, pady=5)

    def open_image(self):
        """Open an image file"""
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if image_path:
            self.image_path = image_path
            self.original_image = cv2.imread(image_path)
            
            # Convert BGR to RGB for display
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # Clear previous boxes
            self.boxes = []
            self.points = []
            
            # Reset zoom
            self.image_scale = 1.0
            
            # Display the image
            self.update_display()
            
            self.status_var.set(f"Opened: {os.path.basename(image_path)}")

    def update_display(self):
        """Update the display with the current image and boxes"""
        if self.original_image is None:
            return
        
        # Create a copy of the original image to draw on
        display_img = self.original_image.copy()
        
        # Draw all saved boxes
        for box in self.boxes:
            points = box["points"]
            color = box["color"]
            thickness = box["thickness"]
            
            # Convert color name to BGR
            color_bgr = self.color_name_to_bgr(color)
            
            # Draw the polygon
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(display_img, [pts], True, color_bgr, thickness)
            
            # Draw the points
            for pt in points:
                cv2.circle(display_img, (int(pt[0]), int(pt[1])), 3, color_bgr, -1)
        
        # Draw the current points being added
        if len(self.points) > 0:
            # Convert color name to BGR
            color_bgr = self.color_name_to_bgr(self.color_var.get())
            thickness = self.thickness_var.get()
            
            # Draw lines between points
            for i in range(len(self.points) - 1):
                pt1 = (int(self.points[i][0]), int(self.points[i][1]))
                pt2 = (int(self.points[i+1][0]), int(self.points[i+1][1]))
                cv2.line(display_img, pt1, pt2, color_bgr, thickness)
            
            # If we have more than 1 point, connect the last point to the first
            if len(self.points) > 1:
                pt1 = (int(self.points[-1][0]), int(self.points[-1][1]))
                pt2 = (int(self.points[0][0]), int(self.points[0][1]))
                cv2.line(display_img, pt1, pt2, color_bgr, thickness)
            
            # Draw the points
            for pt in self.points:
                cv2.circle(display_img, (int(pt[0]), int(pt[1])), 3, color_bgr, -1)
        
        # Apply scaling
        if self.image_scale != 1.0:
            h, w = display_img.shape[:2]
            new_h, new_w = int(h * self.image_scale), int(w * self.image_scale)
            display_img = cv2.resize(display_img, (new_w, new_h))
        
        # Convert to PhotoImage for Tkinter
        self.display_image = Image.fromarray(display_img)
        self.photo_image = ImageTk.PhotoImage(self.display_image)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo_image, anchor=tk.NW)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def color_name_to_bgr(self, color_name):
        """Convert color name to BGR tuple"""
        color_map = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "magenta": (255, 0, 255),
            "cyan": (255, 255, 0),
            "orange": (0, 165, 255),
            "purple": (128, 0, 128)
        }
        return color_map.get(color_name, (0, 0, 255))  # Default to red if not found

    def on_click(self, event):
        """Handle mouse click events"""
        if self.original_image is None:
            return
        
        # Get canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Convert to image coordinates (accounting for scaling)
        img_x = canvas_x / self.image_scale
        img_y = canvas_y / self.image_scale
        
        # Add point to current box
        self.points.append((img_x, img_y))
        self.status_var.set(f"Added point {len(self.points)} at ({int(img_x)}, {int(img_y)})")
        
        # If we have 4 points, complete the box
        if len(self.points) == 4:
            self.add_box()
        
        self.update_display()

    def add_box(self):
        """Add the current points as a box"""
        if len(self.points) >= 3:  # Need at least 3 points for a polygon
            box = {
                "points": self.points.copy(),
                "color": self.color_var.get(),
                "thickness": self.thickness_var.get()
            }
            self.boxes.append(box)
            self.points = []  # Clear current points
            self.status_var.set(f"Added box {len(self.boxes)}")

    def on_mouse_move(self, event):
        """Handle mouse movement"""
        if self.original_image is None:
            return
        
        # Get canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Convert to image coordinates (accounting for scaling)
        img_x = canvas_x / self.image_scale
        img_y = canvas_y / self.image_scale
        
        if 0 <= img_x < self.original_image.shape[1] and 0 <= img_y < self.original_image.shape[0]:
            self.status_var.set(f"Position: ({int(img_x)}, {int(img_y)}) | Points: {len(self.points)}/4")

    def save_config(self):
        """Save the bounding boxes to a config file"""
        if not self.boxes:
            messagebox.showwarning("Warning", "No bounding boxes to save.")
            return
        
        if not self.image_path:
            messagebox.showwarning("Warning", "No image loaded.")
            return
        
        # Default filename based on image name
        default_name = os.path.splitext(os.path.basename(self.image_path))[0] + "_boxes.json"
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            title="Save Config File",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON Files", "*.json")]
        )
        
        if not file_path:
            return
        
        # Create config dictionary
        config = {
            "image_path": self.image_path,
            "boxes": self.boxes
        }
        
        # Save to JSON file
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        self.config_file = file_path
        self.status_var.set(f"Saved config to: {os.path.basename(file_path)}")

    def load_config(self):
        """Load bounding boxes from a config file"""
        file_path = filedialog.askopenfilename(
            title="Load Config File",
            filetypes=[("JSON Files", "*.json")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            # Check if image path exists
            image_path = config.get("image_path", "")
            if not os.path.exists(image_path):
                response = messagebox.askyesno(
                    "Image Not Found",
                    f"The image file specified in the config ({os.path.basename(image_path)}) was not found.\n"
                    "Would you like to locate it manually?"
                )
                
                if response:
                    image_path = filedialog.askopenfilename(
                        title="Select Image File",
                        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")]
                    )
                    
                    if not image_path:
                        return
                else:
                    return
            
            # Load the image
            self.image_path = image_path
            self.original_image = cv2.imread(image_path)
            
            # Convert BGR to RGB for display
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # Set the boxes
            self.boxes = config.get("boxes", [])
            self.points = []
            
            # Reset zoom
            self.image_scale = 1.0
            
            # Update display
            self.update_display()
            
            self.config_file = file_path
            self.status_var.set(f"Loaded config: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config file: {str(e)}")

    def clear_all_boxes(self):
        """Clear all bounding boxes"""
        if self.boxes:
            if messagebox.askyesno("Confirm", "Are you sure you want to clear all boxes?"):
                self.boxes = []
                self.points = []
                self.update_display()
                self.status_var.set("Cleared all boxes")

    def delete_last_box(self):
        """Delete the last drawn box"""
        if self.points:
            self.points = []
            self.status_var.set("Cleared current points")
        elif self.boxes:
            self.boxes.pop()
            self.status_var.set(f"Deleted last box. {len(self.boxes)} remaining.")
        
        self.update_display()

    def zoom_in(self):
        """Zoom in on the image"""
        if self.original_image is not None:
            self.image_scale *= 1.2
            self.update_display()
            self.status_var.set(f"Zoom: {self.image_scale:.1f}x")

    def zoom_out(self):
        """Zoom out from the image"""
        if self.original_image is not None:
            self.image_scale /= 1.2
            if self.image_scale < 0.1:
                self.image_scale = 0.1
            self.update_display()
            self.status_var.set(f"Zoom: {self.image_scale:.1f}x")

    def reset_zoom(self):
        """Reset zoom to original size"""
        if self.original_image is not None:
            self.image_scale = 1.0
            self.update_display()
            self.status_var.set("Zoom reset to 1.0x")

    def show_instructions(self):
        """Show instructions for using the tool"""
        instructions = """
        Oriented Bounding Box Tool Instructions:
        
        1. Open an image using File > Open Image or the Open Image button.
        2. Click on the image to place the 4 corners of your bounding box.
        3. After placing 4 points, the box will be automatically completed.
        4. Use the toolbar to change the color and thickness of the next box.
        5. Save your bounding boxes using File > Save Config.
        6. Load previously saved boxes using File > Load Config.
        7. Use Ctrl+S for saving and Ctrl+O for opening images.
        
        Keyboard Shortcuts:
        - Ctrl+S: Save Config
        - Ctrl+O: Open Image
        - Ctrl+L: Load Config
        
        Notes:
        - You can draw any number of oriented bounding boxes.
        - Each box requires exactly 4 points.
        - The configuration is saved in JSON format.
        """
        messagebox.showinfo("Instructions", instructions)

    def show_about(self):
        """Show about information"""
        about_text = """
        Oriented Bounding Box Tool
        
        A simple tool for drawing oriented bounding boxes on images
        and saving them to a configuration file.
        
        Features:
        - Multiple oriented bounding boxes
        - Customizable colors and line thickness
        - Save and load configurations
        - Image zooming
        """
        messagebox.showinfo("About", about_text)


def main():
    root = tk.Tk()
    root.geometry("1000x700")
    app = OrientedBBoxTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
