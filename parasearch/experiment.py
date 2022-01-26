from dataclasses import dataclass, asdict
import datetime
import os
import json

@dataclass
class BenchSetting:
    updater: str = ""
    problem_name: str = ""
    instance_name: str = ""
    mathematical_model: dict = {}
    ph_value: dict = {}
    opt_value: float = -1
    multipliers: dict = {}

@dataclass
class BenchResult:
    penalties: dict = {}
    raw_response: dict = {}


class Experiment:
    log_filename = "log.json"

    def __init__(self, updater=None, result_dir="./Results") -> None:
        self.updater = updater
        self.setting = BenchSetting()
        self.results = BenchResult()
        self.evaluation_metrics = None
        self.datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = result_dir
        self.log_dir = f"{result_dir}/{self.datetime}/logs"
        self.img_dir = f"{result_dir}/{self.datetime}/imgs"
        self.table_dir = f"{result_dir}/{self.datetime}/tables"
    
    def save(self):
        instance_name = self.setting.instance_name
        save_dir = f"{self.log_dir}/{instance_name}"
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{save_dir}/{self.log_filename}"
        save_obj = {
            "date": str(self.datetime),
            "setting": asdict(self.setting),
            "results": asdict(self.results),
        }
        with open(filename, "w") as f:
            json.dump(save_obj, f)
        
    @classmethod
    def load(cls, filename: str) -> 'Experiment':
        """load date
        saveで保存した結果をそのままloadする.
        Returns:
            Experiment: loaded Experiment object.
        """
        with open(filename, "r") as f:
            data = json.load(f)
        
        date = data['date']
        setting = data['setting']
        results = data['results']

        obj = Experiment()
        obj.datetime = date
        obj.setting = BenchSetting(**setting)
        obj.results = BenchResult(**results)

        return obj

