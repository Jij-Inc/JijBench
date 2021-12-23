from abc import abstractmethod
import json
import glob
import os
import numpy as np
from collections import defaultdict

def make_dist_matrix(coordinates):
    n = len(coordinates)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        x1, y1 = coordinates[i]
        for j in range(n):
            x2, y2 = coordinates[j]
            d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
            dist[i][j] = d2**0.5
    return dist

files = glob.glob("TSPLIB95/tsp/*")
opt_tours = defaultdict(list)
for file in files:
    if "opt.tour" not in file: continue
    with open(file, "r") as f:
        problem = f.read()
    problem = problem.split("\t")

    opt_tour = []
    for i in problem:
        data = i.split()
        ind = data.index("TOUR_SECTION")
        for j in data[ind+1:]:
            if j == "-1" or j == "EOF":
                break
            opt_tour.append(int(j))
    problem_name = file.replace("TSPLIB95/tsp/", "")
    problem_name = problem_name.replace(".opt.tour", "")
    opt_tours[problem_name] = opt_tour

with open("TSPLIB95/tsp/bestSolutions.txt", "r") as f:
    opt_file = f.read()
opt_file = opt_file.split()
opt_dict = {}
for i in range(0, len(opt_file)-2, 3):
    opt_dict[opt_file[i]] = int(opt_file[i+2])

for file in files:
    if file == "TSPLIB95/tsp/bestSolutions.txt": continue
    if "opt.tour" in file: continue
    f = open(file, "r")
    flag = False
    coordinates = []
    while True:
        if data == "": break
        data = f.readline()

        if flag:
            coord = data.split()
            if coord:
                if coord[0] == "EOF":
                    flag = False
                else:
                    x, y = float(coord[1]), float(coord[2])
                    coordinates.append([x, y])
        if "NODE_COORD_SECTION" in data:
            flag = True
    f.close()
    dist = make_dist_matrix(coordinates)
    N = len(dist)
    problem_dir = "/".join(file.split("/")[:-1])
    save_dir = "Instances/TSP/" + problem_dir

    file_name = file.split("/")[-1]
    file_name = file_name.split(".")[:-1]
    opt_value = opt_dict[file_name[0]]
    opt_tour = opt_tours[file_name[0]]
    file_name = "_".join(file_name)
    file_name = file_name + ".json"
    if N >= 500: continue
    ph_value = {"N": len(dist), "dist": dist, "opt_value": opt_value, "opt_tour": opt_tour}
    print(ph_value)

    if not os.path.exists("/app/ParameterSearch/ParameterSearch/" + save_dir):
        os.makedirs("/app/ParameterSearch/ParameterSearch/" + save_dir)

    with open(save_dir + "/" + file_name, "w") as g:
        json.dump(ph_value, g)

