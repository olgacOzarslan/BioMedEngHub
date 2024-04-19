from tkinter import *
from tkinter import filedialog
from PIL import ImageTk,Image
from ultralytics import YOLO
import cv2
from cv2 import aruco
import numpy as np
import os
import uuid
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import subprocess

global image_with_predictions

def append_message(msg = ""):
    global info_text_field
    info_text_field.insert(END, msg + "\n")
    info_text_field.see(END)  # Scroll to the end
        
def open_file():
    global sample_image_canvas, preprocessed_image_canvas, preprocessed_image_tk, sample_image_id, sample_image_tk
    initial_dir = "~/Desktop"
    filename = filedialog.askopenfilename(initialdir=initial_dir)
    if filename:
        # Preprocess image in here
        pil_image = Image.open(filename)
        pil_image = pil_image.resize((256,256))
        #pil_image = pil_image.rotate(270)
        sample_image_tk = ImageTk.PhotoImage(pil_image)
        sample_image_canvas.itemconfig(sample_image_id, image = sample_image_tk)
        preprocess(filename)
        predict(preprocessed_image)
def predict(image):
    global bar_canvas, fig
    global model
    global predict_image_canvas
    global preprocessed_image
    global predict_image_id
    global predict_image_tk
    global image_with_predictions
    results = model(preprocessed_image)
    image_with_predictions = results[0].plot()
    img_with_predictions_rgb = cv2.cvtColor(image_with_predictions, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_with_predictions_rgb) 
    predict_image_tk = ImageTk.PhotoImage(pil_image)
    predict_image_canvas.itemconfig(preprocessed_image_id, image = predict_image_tk)
    # get predicted labels
    labels= {"5<ph<7": 0,
              "7<ph<8": 0,
              "8<ph<10":0,
              "sc<10":0,
              "sc>10":0
            }
    
    for _ in results[0].boxes.cls:
        labels[results[0].names[int(_)]] += 1
    
    fig.clear()
    plt.bar(labels.keys(), labels.values())
    plt.xticks(fontsize=4, rotation=45)  # Set x-tick label font size and rotation
    plt.yticks(fontsize=4)  # Set y-tick label font size
    # Additional customizations (optional)
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', which='both', alpha=0.7)  # Grid lines
    plt.tight_layout()  # Adjust spacing to avoid overlapping elements
    bar_canvas.draw()

def save_results():
    initial_dir = "~/Desktop/"
    folderpath = filedialog.askdirectory(initialdir=initial_dir)
    if folderpath:
        unique_id = str(uuid.uuid4())[:8]
        save_preprocessed_image(folderpath, unique_id)
        save_yolo_image(folderpath, unique_id)
        save_plot(folderpath, unique_id)
        append_message(f"Sucess. Plots saved at ---> {folderpath}")
    # open result folder

def save_plot(filepath, unique_id):
    global fig
    path = os.path.join(filepath, f"results_{unique_id}.png")
    fig.savefig(path)

def save_preprocessed_image(filepath, unique_id):
    global image_with_predictions
    path = os.path.join(filepath, f"preprocessed_{unique_id}.jpg")
    cv2.imwrite(path, preprocessed_image)

def save_yolo_image(filepath, unique_id):
    global preprocessed_image
    path = os.path.join(filepath, f"YOLO_{unique_id}.jpg")
    cv2.imwrite(path, image_with_predictions)


def template_matching(input_image):
    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    reference_image = cv2.imread("references/base.JPG", cv2.IMREAD_GRAYSCALE)
    result = cv2.matchTemplate(input_gray, reference_image, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    h, w = reference_image.shape
    top_left = max_loc
    bottom_right = (top_left[0]+w, top_left[1] + h)
    cropped_region = input_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cropped_region = cv2.resize(cropped_region, (256, 256))
    return cropped_region
def sort_corners(corners, ids):
    # Assume that ids are [0, 1, 2, 3] for [top-left, top-right, bottom-right, bottom-left]
    # Each id corresponds to a marker, and each marker has 4 corners
    # Calculate the center of each marker and sort them
    markers_center = np.zeros((4,2), dtype=np.float32)
    for i, id in enumerate(ids):
        # Calculate the center point of each marker
        marker_center = np.average(corners[i][0], axis=0)
        markers_center[id[0]] = marker_center

    # If some ids are missing, this will raise a ValueError
    if np.any(np.sum(markers_center, axis=1) == 0):
        raise ValueError('Some corners were not assigned!')

    return markers_center
def extract_roi(image_path, draw_detected_markers=False):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected_img_points = detector.detectMarkers(gray)
    if draw_detected_markers:
        cv2.imwrite("./detected_markers.png", cv2.aruco.drawDetectedMarkers(gray, corners, ids))
    if ids is not None and len(ids) == 4:
        corners = sort_corners(corners, ids)
    else:
        raise ValueError('Detected markers count is not equal to four.')
    dst_pts = np.array([ [600, 800], [0, 800], [600, 0], [0, 0]], dtype="float32")
    M = cv2.getPerspectiveTransform(corners, dst_pts)
    warped_img = cv2.warpPerspective(image, M, (600, 800))
    return warped_img
def preprocess(img_path = ""):
    global preprocessed_image
    global preprocessed_image_name
    global preprocessed_image_id
    global preprocessed_image_canvas
    global preprocessed_image_tk
    img = extract_roi(image_path = img_path)
    preprocessed_image = template_matching(img)
    preprocessed_image_name = img_path.split('/')[-1].split('.')[0]
    # set preprocessed image canvas
    preprocessed_image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(preprocessed_image_rgb) 
    preprocessed_image_tk = ImageTk.PhotoImage(pil_image)
    preprocessed_image_canvas.itemconfig(preprocessed_image_id, image = preprocessed_image_tk)




model = YOLO("last.pt")

window = Tk()
window.title("DXBiotech Lab")

# Create Logo Canvas -- TOP FRAME
logo_canvas_width = 100
logo_canvas_height = 75
logo_canvas = Canvas(window, width=logo_canvas_width, height=logo_canvas_height)
logo = Image.open("misc/logo.jpg")
logo = logo.resize((logo_canvas_width, logo_canvas_height))
logo_tk = ImageTk.PhotoImage(logo)
logo_canvas.create_image(0,0,anchor="nw", image=logo_tk)
logo_canvas.pack()


sample_image_frame = Frame(window, highlightbackground="gray", highlightthickness=1)
preprocessed_image_frame = Frame(window, highlightbackground="gray", highlightthickness=1)
yolo_prediction_frame = Frame(window, highlightbackground= "gray", highlightthickness=1)
bar_plot_frame = Frame(window, highlightbackground= "gray", highlightthickness=1)
button_frame = Frame(window, highlightbackground="gray", highlightthickness=1)

sample_image_frame.pack(side="left", fill="both", expand=True)
preprocessed_image_frame.pack(side="left", fill="both", expand=True)
yolo_prediction_frame.pack(side="left", fill="both", expand=True)
bar_plot_frame.pack(side="left", fill="both", expand=True)
button_frame.pack(side="left", fill="both", expand=True)

# Construct sample image frame
sample_image_label = Label(sample_image_frame, text="Sample Image").pack(padx=2, pady = 2)

sample_canvas_width = 256  # Adjust width as needed
sample_canvas_height = 256  # Adjust height as needed
sample_image_canvas = Canvas(sample_image_frame, width=sample_canvas_width, height=sample_canvas_height)
tmp = Image.open("misc/blank_image.png")
tmp = tmp.resize((sample_canvas_width, sample_canvas_height))
sample_image_tk = ImageTk.PhotoImage(tmp)
sample_image_id = sample_image_canvas.create_image(0, 0, anchor="nw", image=sample_image_tk)
sample_image_canvas.pack(padx=2, pady=2)

# Construct preprocessed image frame

preprocessed_image_label = Label(preprocessed_image_frame, text="Preprocessed Image").pack(padx=2, pady = 2)

preprocessed_canvas_width = 256  # Adjust width as needed
preprocessed_canvas_height = 256  # Adjust height as needed
preprocessed_image_canvas = Canvas(preprocessed_image_frame, width=preprocessed_canvas_width, height=preprocessed_canvas_height)
preprocess_tmp = Image.open("misc/blank_image.png")
preprocess_tmp = preprocess_tmp.resize((preprocessed_canvas_width, preprocessed_canvas_height))
preprocessed_img_tk = ImageTk.PhotoImage(preprocess_tmp)
preprocessed_image_id = preprocessed_image_canvas.create_image(0, 0, anchor="nw", image=preprocessed_img_tk)
preprocessed_image_canvas.pack()

# Construct Yolo prediction frame
predict_image_label = Label(yolo_prediction_frame, text="Prediction Result").pack(padx=2, pady=2)

predict_image_canvas = Canvas(yolo_prediction_frame, width=preprocessed_canvas_width, height=preprocessed_canvas_height)
predict_tmp = Image.open("misc/blank_image.png")
predict_tmp = predict_tmp.resize((preprocessed_canvas_width, preprocessed_canvas_height))
predict_image_tk = ImageTk.PhotoImage(predict_tmp)
predict_image_id = predict_image_canvas.create_image(0, 0, anchor="nw", image=predict_image_tk)
predict_image_canvas.pack()

# Construct Bar plot frame

# Create your Matplotlib plot (replace with your code)
prediction_result_label = Label(bar_plot_frame, text="Class Distributions").pack(padx=2,pady=2)
fig = plt.figure(figsize=(1.7,1.7))
x = ["5<ph<7", "7<ph<8", "8<ph<10", "sc<10", "10<sc"]
y = ["0", "0", "0", "0", "0"]
plt.bar(x, y)  # Create the bar plot
plt.xticks(fontsize=4, rotation=45)  # Set x-tick label font size and rotation
plt.yticks(fontsize=4)  # Set y-tick label font size

# Additional customizations (optional)
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', which='both', alpha=0.7)  # Grid lines
plt.tight_layout()  # Adjust spacing to avoid overlapping elements


bar_canvas = FigureCanvasTkAgg(fig, master=bar_plot_frame)
bar_canvas.get_tk_widget().pack()
#save_file_button = Button(right_frame, text="Save Result", command=save_images).pack(padx=2, pady = 2)
#sample_image_button = Button(bottom_frame, text="Load Sample Image", command=open_file).pack(padx=2, pady = 2)

open_file_button = Button(button_frame, text="Load Sample Image", command=open_file)
open_file_button.pack(padx=2,pady=2)
save_file_button = Button(button_frame, text="Save Results", command=save_results)
save_file_button.pack(padx=2,pady=2)

info_text_field = Text(button_frame, wrap='word', width=20, height=20)
info_text_field.pack(padx=2, pady = 2, expand=True)

window.mainloop()
