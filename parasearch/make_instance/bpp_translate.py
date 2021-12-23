import json
import glob
import os


files = glob.glob("Augmented_Non_IRUP_and_Augmented_IRUP_Instances/Difficult_Instances/ANI/*")
for file in files:
    with open(file, "r") as f:
        problem = f.read()
    
    problem = list(map(int, problem.split("\n")[:-1]))
    ph_value = {"num_items":problem[0], "w":[], "c":problem[1]}
    for weight in problem[2:]:
        ph_value["w"].append(weight)

    problem_name = file.split(".")[0]
    problem_dir = "/".join(problem_name.split("/")[:-1])
    save_dir = "Instances/bin_packing/" + problem_dir
    
    file_name = problem_name.split("/")[-1] + ".json"

    if not os.path.exists("/app/BPPbenchmark/" + save_dir):
        os.makedirs("/app/BPPbenchmark/" + save_dir)

    with open(save_dir + "/" + file_name, "w") as f:
        json.dump(ph_value, f)
