import os 
from matplotlib import pyplot as plt
param = "glu"
title = "110"
current_directory = os.path.dirname(os.path.abspath(__file__))
# Navigate one level up
parent_directory = os.path.dirname(current_directory)

abs_path = os.path.join(parent_directory,f"measurements/{param}")
filename = "13-48-27_glu_5.txt"

file = os.path.join(abs_path,filename)

if os.path.isfile(file):
    with open(file, 'r') as f:
        reds  = []
        greens = []
        blues = []
        data = f.read()
        measurements = data.split('\n')[1].split(';')
        for measurement in measurements:
            try:
                r = float(measurement.split(',')[0])
                g = float(measurement.split(',')[1])
                b = float(measurement.split(',')[2])
                reds.append(r)
                greens.append(g)
                blues.append(b)
            except:
                print(f"red : {r}")
                print(f"green : {g}")
                print(f"blue : {b}")
else:
    print("error while parsing")

red_max = max(reds)
if red_max > 255:
    reds = [(red / red_max)*255 for red in reds]

green_max = max(greens)
if green_max > 255:
    greens = [(green / green_max)*255 for green in greens]

blue_max = max(blues)
if blue_max > 255:
    blues = [(blue / blue_max)*255 for blue in blues]



fig, axs = plt.subplots(1,3)
x = [i for i in range(len(reds))]
axs[0].plot(x, reds, '-r')
axs[1].plot(x, greens,'-g')
axs[2].plot(x,blues, '-b')
fig.show()
user_input = input("Enter something: ")

upper_limit = int(user_input.split(',')[1])
lower_limit = int(user_input.split(',')[0])
reds = reds[lower_limit:upper_limit]
greens = greens[lower_limit:upper_limit]
blues = blues[lower_limit:upper_limit]

# normalize the measurements
red_mean = sum(reds) / len(reds)
#dev = (sum([(x - mean)**2 for x in reds])/len(reds))** 0.5
#reds = [(x-mean)/dev for x in reds]
#print(f"red mean : {mean}, red dev : {dev}")

green_mean = sum(greens) / len(greens)
#dev = (sum([(x - mean)**2 for x in greens])/len(greens))** 0.5
#greens = [(x-mean)/dev for x in greens]
#print(f"green mean : {mean}, green dev : {dev}")

blue_mean = sum(blues) / len(blues)
#dev = (sum([(x - mean)**2 for x in blues])/len(blues))** 0.5
#blues = [(x-mean)/dev for x in blues]
#print(f"blue mean : {mean}, blue dev : {dev}")
# find the point which minimize the distance

centroid = (red_mean, green_mean, blue_mean )
print(centroid)
file_path = os.path.join(parent_directory, f"calibration/{param}.txt")
with open(file_path, 'a') as f:
    f.write(f"{title}_{str(centroid[0])},{str(centroid[1])},{str(centroid[2])}")
