import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from  matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import numpy as np
import serial
import time
import os
import sys
ser = serial.Serial(port='/dev/cu.usbserial-0001', baudrate=115200, timeout=1)
ser.close()

if not ser.is_open:
    ser.open()



# Main application window
root = tk.Tk()
root.title("GUI Application")

labels = ["LEU", "NIT", "URO", "PRO", "pH", "BLO", "SG", "KET", "BIL", "GLU"]

# Create a title label
title_frame = tk.Frame(root)
title_frame.grid(row=0, column=0, columnspan=len(labels), pady=10)  # Added padding
title = tk.Label(title_frame, width=20, text="Operator Readings")
title.pack()

# Create a frame for the labels and input fields
entry_frame = tk.Frame(root)
entry_frame.grid(row=1, column=0, columnspan=len(labels), pady=10)  # Added padding

entries = []

# Place labels and input fields inside the entry_frame
for i, _ in enumerate(labels):
    label = ttk.Label(entry_frame, text=_)
    label.grid(row=0, column=i, padx=5)
    entry = ttk.Entry(entry_frame, width=3)
    entry.grid(row=1, column=i, padx=5, pady=5)
    entries.append(entry)

# Create a frame for the filename label and input field
filename_frame = tk.Frame(root)
filename_frame.grid(row=2, column=0, pady=10)  # Added padding

filename_label = tk.Label(filename_frame, text="filename")
filename_label.grid(row=0, column=0, padx=5, pady=5)

filename_entry = tk.Entry(filename_frame)
filename_entry.grid(row=0, column=1)

def save_measurement():
    global filename_entry
    global entries
    global measurements
    filename = filename_entry.get()
    main_dir = os.path.expanduser("~/Desktop")  # Change this to your desired main directory
    timestr = time.strftime("%Y-%m-%d:%H-%M-%S")
    year = timestr.split(':')[0].split('-')[0]
    month = timestr.split(':')[0].split('-')[1]
    day = timestr.split(':')[0].split('-')[2]
    hour = timestr.split(':')[1]
    sub_dir = f"measurements/{year}/{month}/{day}"
    complete_path = os.path.join(main_dir, sub_dir)
    os.makedirs(complete_path, exist_ok=True)
    filename_with_extension = f"{hour}_{filename}.txt"
    complete_path = os.path.join(complete_path, filename_with_extension)
    #plt.savefig(f"figures/{filename}.png")
    #img.save(f"images/{filename}.png")
    with open(complete_path, 'w') as file:
        txt = ""
        for i, entry in enumerate(entries):
            txt += entry.get().strip()
            if i != len(entries) - 1:
                txt += '$'
        txt += "\n"
        for meas in measurements:
            txt += f"{str(meas[0])}, {str(meas[1])}, {str(meas[2])};"
        file.write(txt)

    append_text(f"Measurement successfully saved to --> {complete_path}")    
    
filename_button = tk.Button(filename_frame, text="Save Measurement", command = save_measurement, width=15)
filename_button.grid(row=0, column=2)

# Create a frame for device operation buttons
device_operation_frame = tk.Frame(root)
device_operation_frame.grid(row=3, column=0, pady=10)  # Added padding

def open_device_cartride():
    global ser
    timeout = 10
    if not ser.is_open:
        ser.open()
    ser.reset_output_buffer()
    ser.reset_input_buffer()
    ser.write("Open".encode())
    append_text("Standby... waiting for cartride to open")
    dt = 0
    current_time = time.time()
    previous_time = time.time()
    last_received = ""
    while dt < timeout and "Done" not in last_received:
        current_time = time.time()
        dt = current_time - previous_time
        if ser.in_waiting:
            previous_time = time.time()
            last_received = ser.readline().decode().strip()
            append_text(last_received)
    append_text("Ready")
open_button = tk.Button(device_operation_frame, text="Open", command=open_device_cartride, width=10)
open_button.grid(row=0, column=0)

def close_device_cartride():
    global ser
    timeout = 10
    if not ser.is_open:
        ser.open()
    ser.reset_input_buffer()    
    ser.reset_output_buffer()
    ser.write("Close".encode())
    append_text("Standby... waiting for cartride to close")
    dt = 0
    current_time = time.time()
    previous_time = time.time()
    last_received = ""
    while dt < timeout and "Done" not in last_received:
        current_time = time.time()
        dt = current_time - previous_time
        if ser.in_waiting:
            previous_time = time.time()
            last_received = ser.readline().decode().strip()
            append_text(last_received)
    append_text("Ready")
close_button = tk.Button(device_operation_frame, text="Close", command = close_device_cartride, width=10)
close_button.grid(row=0, column=1)

measurements = []
def measure_functionality():
    global measurements  # Declare it as global to modify it
    global ser
    #ser.reset_input_buffer()
    #ser.reset_output_buffer()
    ser.write("Start".encode())
    measurements = []  # Clear the previous measurements if any
    timeout = 10  # Timeout in seconds for demonstration
    # Dummy serial port for demonstration; please replace with your actual port
    previous = time.time()
    dt = 0
    raw = ""
    while dt < timeout and "Done" not in raw:
        current = time.time()
        dt = current - previous
        if ser.in_waiting:
            raw = ser.readline().decode().strip()
            append_text(raw)
            previous = current
            data = raw.split(',')
            if len(data) == 3:
                rgb = [int(float(_) * 255) for _ in data]
                measurements.append(rgb)
    # Create matplotlib figure and draw plots
    fig = Figure(figsize=(8, 4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    ax1.plot([x[0] for x in measurements], 'r')
    ax1.set_title("Red")
    ax2.plot([x[1] for x in measurements], 'g')
    ax2.set_title("Green")
    ax3.plot([x[2] for x in measurements], 'b')
    ax3.set_title("Blue")
    
    plt.savefig("measurements.png")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=5, column=0)
    
    # Create an image using PIL
    img_array = np.array(measurements, dtype=np.uint8)
    img_array = np.repeat(img_array[np.newaxis, :, :], 200, axis=0)
    img = Image.fromarray(img_array, 'RGB')
    img.save("roi_extraction.png")
    # Convert PIL image to ImageTk format and display
    img_tk = ImageTk.PhotoImage(img)
    img_label = tk.Label(root, image=img_tk)
    img_label.image = img_tk
    img_label.grid(row=6, column=0)
measure_button = tk.Button(device_operation_frame, text="Measure", command=measure_functionality, width=10)
measure_button.grid(row=0, column=2)

def calibrate_functionality():
    global ser
    timeout = 10
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.1)
    ser.write("Calibrate".encode())
    time.sleep(1)
    append_text("Calibrating...")
    current_time = time.time()
    previous_time = time.time()
    dt = 0
    last_received = ""
    while dt < timeout and "Done" not in last_received:
        current_time = time.time() 
        dt = current_time - previous_time
        if ser.in_waiting:
            previous_time = time.time()
            last_received = ser.readline().decode().strip()
            append_text(last_received)
            root.mainloop()
    append_text("Calibrated...")
    append_text("Ready")
calibrate_button = tk.Button(device_operation_frame, text="Calibrate",command = calibrate_functionality, width=10)
calibrate_button.grid(row=0, column=3)

# Create a frame for the text field and clear button
text_frame = tk.Frame(root)
text_frame.grid(row=4, column=0, pady=10)  # Added padding

# Create a Text widget
text_field = tk.Text(text_frame, wrap='word', width=50, height=10)
text_field.grid(row=0, column=0)  # Placed inside text_frame

# Function to append text to the Text widget
def append_text(message):
    text_field.insert(tk.END, message + "\n")
    text_field.see(tk.END)  # Scroll to the end

# Function to clear the Text widget
def clear_text():
    text_field.delete(1.0, tk.END)

# Create a Clear button
clear_button = tk.Button(text_frame, text="Clear", command=clear_text)
clear_button.grid(row=0, column=1)  # Placed inside text_frame

# Test the append_text function
append_text("Ready. Let's start!")

root.mainloop()




