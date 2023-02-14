import jijmodeling as jm

from jijbench.experiment.experiment import Experiment
from jijbench.solver.base import Return
from jijbench.functions.factory import RecordFactory


def construct_experiment_from_sampleset(sampleset: jm.SampleSet) -> Experiment:
    experiment = Experiment(autosave=False)
    factory = RecordFactory()
    ret = [Return(data=sampleset, name="")]
    record = factory(ret)
    experiment.append(record)
    return experiment
