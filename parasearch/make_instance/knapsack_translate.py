import json
import glob
import os


files = glob.glob("instances_01_KP/large_scale/*")
for file in files:
    with open(file, "r") as f:
        problem = f.read()

    num_items, capacity = problem.split("\n")[0].split()
    num_items = int(num_items)
    capacity = int(capacity)

    items = problem.split("\n")[1:-2]

    weights = []
    values = []
    for s in items:
        w, v = s.split()
        w = float(w)
        v = float(v)
        weights.append(w)
        values.append(v)

    ph_value = {"weights": weights, "values": values, "num_items": num_items, "capacity": capacity} 

    problem_dir = "/".join(file.split("/")[:-1])
    save_dir = "Instances/knapsack/" + problem_dir
    
    file_name = file.split("/")[-1] + ".json"

    if not os.path.exists("/app/ParameterSearch/ParameterSearch/" + save_dir):    
        os.makedirs("/app/ParameterSearch/ParameterSearch/" + save_dir)

    with open(save_dir + "/" + file_name, "w") as f:
        json.dump(ph_value, f)

