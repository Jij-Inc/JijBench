from abc import abstractmethod
import json
import glob
import os
import numpy as np


def make_dist_matrix(coordinates):
    n = len(coordinates)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        x1, y1 = coordinates[i]
        for j in range(n):
            x2, y2 = coordinates[j]
            d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
            dist[i][j] = int(d2**0.5)
    return dist


files = glob.glob("TSPLIB95/atsp/*")
with open("TSPLIB95/atsp/bestSolutions.txt", "r") as f:
    opt_file = f.read()
opt_file = opt_file.split()
opt_dict = {}
for i in range(0, len(opt_file)-1, 2):
    opt_dict[opt_file[i][:-1]] = int(opt_file[i+1])


for file in files:
    if file == "TSPLIB95/atsp/bestSolutions.txt": continue
    with open(file, "r") as f:
        problem = f.read()
    problem = problem.split("\n")
    N = int(problem[3].split()[-1])


    dist = []
    for i in problem[7:]:
        data = i.split()
        for d in data:
            if d == "EOF": break
            dist.append(int(d))

    dist = np.reshape(dist, (N, N))
    ndist = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            ndist[i][j] = int(dist[i][j])
    problem_dir = "/".join(file.split("/")[:-1])
    save_dir = "Instances/TSP/" + problem_dir

    file_name = file.split("/")[-1]
    file_name = file_name.split(".")[:-1]
    opt_value = opt_dict[file_name[0]]
    file_name = "_".join(file_name)
    file_name = file_name + ".json"
    ph_value = {"N": N, "dist": ndist, "opt_value": opt_value}

    if not os.path.exists("/app/ParameterSearch/ParameterSearch/" + save_dir):
        os.makedirs("/app/ParameterSearch/ParameterSearch/" + save_dir)

    with open(save_dir + "/" + file_name, "w") as f:
        json.dump(ph_value, f)

