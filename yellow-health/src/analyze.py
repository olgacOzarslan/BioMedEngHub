import os
import math
import sys

def get_cmetry_result(parameter = "", rgb = []):
    scores = {}
    cfg_file = [file for file in os.listdir("calibration/") if file.endswith(".txt") and file.split('_')[1] == parameter]
    with open(os.path.join("calibration", cfg_file)) as f:
        for line in f:
            title = line.split('_')[0] # ++, --, +-
            centroid = [float(x) for x in line.split('_')[1].split(',')]
            score = find_distance(rgb, centroid)
            scores[title] = score
    cmetry_result = min(scores, key=scores.get)
    return cmetry_result

def find_distance(rgb = [], centroid = []):
    return math.sqrt(sum([(x1-x2)**2 for x1,x2 in zip(rgb,centroid)]))

def concatenate_rgb_measurements(measurement = []):
    print("bummm cooncatena ediyorumm")
    return []

if len(sys.argv) != 3:
    print("Usage: python3 analyze.py filepath/name")
    sys.exit(1)





filepath = sys.argv[1]
parameters = ["LEU", "NIT", "URO", "PRO","pH", "BLO", "SG", "KET", "BIL", "GLU"]
rgbs = []
with open(filepath) as file:
    for i,line in enumerate(file):
        if i != 0:
            for rgb in line.split(';'):
                rgb = [float(x) for x in rgb.split(',')]
                rgbs.append(rgb)    
meaasurement = concatenate_rgb_measurements(rgbs)
tmp = []
for parameter, rgb in zip(parameters, rgbs):
    tmp.append(get_cmetry_result(parameter, rgb))
results = {param:result for param, result in zip(parameters, tmp)}
print(results)