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
import scipy.signal as signal
import math


try:
    ser = serial.Serial(port='/dev/cu.usbserial-0001', baudrate=115200, timeout=1)
    if not ser.is_open:
        ser.open()
except:
    print("serial error")
    #sys.exit(1)

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

def open_device_cartridge():
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

def close_device_cartridge():
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

def measure_functionality():
    global measurements
    global ser
    global obtained_rgb_values
    global fig
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.write("Start".encode())
    measurements = []

    timeout = 10
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
    reds = [x[0] for x in measurements]
    greens = [x[1] for x in measurements]
    blues = [x[2] for x in measurements]

    red_max = max(reds)
    if red_max > 255:
        reds = [(red / red_max)*255 for red in reds]

    green_max = max(greens)
    if green_max > 255:
        greens = [(green / green_max)*255 for green in greens]

    blue_max = max(blues)
    if blue_max > 255:
        blues = [(blue / blue_max)*255 for blue in blues]
    
    measurements = []
    for i in range(len(reds)):
        rgb = [reds[i], greens[i], blues[i]]
        measurements.append(rgb)
    # Create matplotlib figure and draw plots
    fig = Figure(figsize=(8, 4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    red_signal = [x[0] for x in measurements]
    ax1.plot(red_signal, 'r')
    ax1.set_title("Red")

    ax2.plot([x[1] for x in measurements], 'g')
    ax2.set_title("Green")

    ax3.plot([x[2] for x in measurements], 'b')
    ax3.set_title("Blue")

    # Mark local maxima on the red signal
    local_maxima_indices = signal.find_peaks(red_signal, prominence=10)[0].tolist()
    #will removed in future

    if len(local_maxima_indices) != 10:
        local_maxima_indices.append(len(red_signal)-1)
    local_maxima_values = [red_signal[i] for i in local_maxima_indices]

    obtained_rgb_values = [measurements[i] for i in local_maxima_indices]


    ax1.plot(local_maxima_indices, local_maxima_values, 'go', markersize=5, label='Local Maxima')

    ax1.legend()

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=5, column=0)

    
    #Create an image using PIL
    #img_array = np.array(measurements, dtype=np.uint8)
    #img_array = np.repeat(img_array[np.newaxis, :, :], height, axis=0)
    #img = Image.fromarray(img_array, 'RGB')

    width = len(measurements)
    height = 50
    # Create an empty image
    img_array = np.empty((height, width, 3), dtype=np.uint8)
    # Fill the image with RGB values
    for i in range(height):
        img_array[i, :, :] = measurements

    # Create an image using PIL
    img = Image.fromarray(img_array, 'RGB')
    # Convert PIL image to ImageTk format and display
    img_tk = ImageTk.PhotoImage(img)
    img_label = tk.Label(root, image=img_tk)
    img_label.image = img_tk
    img_label.grid(row=6, column=0)

    obtained_rgb_images = []  # Create an empty list to store individual images

    width = int(50)
    height = int(50)

    for rgb in obtained_rgb_values:
        # Create a NumPy array filled with the RGB color
        img_array = np.full((height, width, 3), rgb, dtype=np.uint8)
        individual_img = Image.fromarray(img_array, 'RGB')
        obtained_rgb_images.append(individual_img)

    # Concatenate the individual images horizontally to form a single image
    concat_image = Image.new('RGB', (width * len(obtained_rgb_images), height))
    x_offset = 0
    for img in obtained_rgb_images:
        concat_image.paste(img, (x_offset, 0))
        x_offset += width

    # Convert PIL image to ImageTk format and display
    concat_img_tk = ImageTk.PhotoImage(concat_image)
    concat_img_label = tk.Label(root, image=concat_img_tk)
    concat_img_label.image = concat_img_tk
    concat_img_label.grid(row=6, column=1, padx=10, pady=10)


    
    global labels
    scores = {}
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # Navigate one level up
    parent_directory = os.path.dirname(current_directory)
    for i,label in enumerate(labels):
        scores = {}
        path = os.path.join(parent_directory, f"calibration/{label}.txt")
        #cfg_file = [file for file in os.listdir(path) if file.endswith(".txt") and file.split('_')[1] == label]
        with open(path, 'r') as f:
            for line in f:
                title = line.split('_')[0] # ++, --, +-
                centroid = [float(x) for x in line.split('_')[1].split(',')]
                score = find_distance(obtained_rgb_values[i], centroid)
                scores[title] = score
        cmetry_result = min(scores, key=scores.get)
        text = f"{label}:{cmetry_result}"
        append_text(text)
    
def find_distance(rgb = [], centroid = []):
    return math.sqrt(sum([(x1-x2)**2 for x1,x2 in zip(rgb,centroid)]))
    
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

def append_text(message):
    text_field.insert(tk.END, message + "\n")
    text_field.see(tk.END)

def clear_text():
    text_field.delete(1.0, tk.END)



root = tk.Tk()
root.title("GUI Application")

obtained_rgb_values = []

labels = ["leu", "nit", "uro", "pro", "ph", "blo", "sg", "ket", "bil", "glu"]

# Create a title label
title_frame = tk.Frame(root)
title_frame.grid(row=0, column=0, columnspan=len(labels), pady=10)
title = tk.Label(title_frame, width=20, text="Operator Readings")
title.pack()

# Create a frame for the labels and input fields
entry_frame = tk.Frame(root)
entry_frame.grid(row=1, column=0, columnspan=len(labels), pady=10)

entries = []
results = []

for i, label in enumerate(labels):
    label = ttk.Label(entry_frame, text=label)
    label.grid(row=0, column=i, padx=5)
    entry = ttk.Entry(entry_frame, width=3)
    entry.grid(row=1, column=i, padx=5, pady=5)
    entries.append(entry)


# Create a frame for the filename label and input field
filename_frame = tk.Frame(root)
filename_frame.grid(row=2, column=0, pady=10)

filename_label = tk.Label(filename_frame, text="filename")
filename_label.grid(row=0, column=0, padx=5, pady=5)

filename_entry = tk.Entry(filename_frame)
filename_entry.grid(row=0, column=1)

filename_button = tk.Button(filename_frame, text="Save Measurement", command=save_measurement, width=15)
filename_button.grid(row=0, column=2)

# Create a frame for device operation buttons
device_operation_frame = tk.Frame(root)
device_operation_frame.grid(row=3, column=0, pady=10)

open_button = tk.Button(device_operation_frame, text="Open", command=open_device_cartridge, width=10)
open_button.grid(row=0, column=0)

close_button = tk.Button(device_operation_frame, text="Close", command=close_device_cartridge, width=10)
close_button.grid(row=0, column=1)

measure_button = tk.Button(device_operation_frame, text="Measure", command=measure_functionality, width=10)
measure_button.grid(row=0, column=2)

calibrate_button = tk.Button(device_operation_frame, text="Calibrate", command=calibrate_functionality, width=10)
calibrate_button.grid(row=0, column=3)

# Create a Text widget
text_field = tk.Text(root, wrap='word', width=50, height=10)
text_field.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")

# Create a vertical scrollbar
#scrollbar = tk.Scrollbar(root, orient="vertical", command=text_field.yview)
#scrollbar.grid(row=4, column=1, sticky="ns")
#text_field.config(yscrollcommand=scrollbar.set)

# Test the append_text function
append_text("Ready. Let's start!")

# Allow the Text widget to expand with the window
root.grid_rowconfigure(4, weight=1)
root.grid_columnconfigure(0, weight=1)

root.mainloop()
