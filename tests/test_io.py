import dill
import jijbench as jb
import pandas as pd

from icecream import ic


SAVEDIR = "./tests/sample"


def test_save_artifact_with_mode_w():
    experiment_name = "test"
    a = jb.Artifact({0: {"a": jb.Number(1, "num")}})
    jb.save(a, "test", savedir=SAVEDIR)

    with open(f"{SAVEDIR}/{experiment_name}/artifact.dill", "rb") as f:
        res = dill.load(f)

    assert res == a


def test_save_artifact_with_mode_a():
    experiment_name = "test"
    a0 = jb.Artifact({0: {"a": jb.Number(1, "num")}})
    jb.save(a0, experiment_name, savedir=SAVEDIR)

    a1 = jb.Artifact({1: {"a": jb.Number(1, "num")}})
    jb.save(a1, experiment_name, savedir=SAVEDIR, mode="a")

    with open(f"{SAVEDIR}/{experiment_name}/artifact.dill", "rb") as f:
        res = dill.load(f)

    assert res.data[0] == a0.data[0]
    assert res.data[1] == a1.data[1]


def test_load_artifact():
    experiment_name = "test"
    a0 = jb.Artifact({0: {"a": jb.Number(1, "num")}})
    jb.save(a0, experiment_name, savedir=SAVEDIR)

    a1 = jb.Artifact({1: {"a": jb.Number(1, "num")}})
    jb.save(a1, experiment_name, savedir=SAVEDIR, mode="a")

    res = jb.load(experiment_name, savedir=SAVEDIR, return_type="Artifact")
    assert res.data[0] == a0.data[0]
    assert res.data[1] == a1.data[1]


def test_save_table_with_mode_w():
    experiment_name = "test"
    t = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["a"]))
    jb.save(t, experiment_name, savedir=SAVEDIR, mode="w")

    res = pd.read_csv(f"{SAVEDIR}/{experiment_name}/table.csv", index_col=0)
    with open(f"{SAVEDIR}/{experiment_name}/meta.dill", "rb") as f:
        meta = dill.load(f)

    assert res.loc["a", "0"] == 1
    assert meta["dtype"][0] == jb.Number


def test_save_table_with_mode_a():
    experiment_name = "test"
    t0 = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["a"]))
    jb.save(t0, experiment_name, savedir=SAVEDIR)

    t1 = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["b"]))
    jb.save(t1, experiment_name, savedir=SAVEDIR, mode="a")

    res = pd.read_csv(f"{SAVEDIR}/{experiment_name}/table.csv", index_col=0)
    with open(f"{SAVEDIR}/{experiment_name}/meta.dill", "rb") as f:
        meta = dill.load(f)

    assert res.loc["a", "0"] == 1
    assert res.loc["b", "0"] == 1
    assert meta["dtype"][0] == jb.Number


def test_load_table():
    experiment_name = "test"
    t0 = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["a"]))
    jb.save(t0, experiment_name, savedir=SAVEDIR)

    t1 = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["b"]))
    jb.save(t1, experiment_name, savedir=SAVEDIR, mode="a")

    res = jb.load(experiment_name, savedir=SAVEDIR, return_type="Table")
    assert res.data.loc["a", 0] == t0.data.loc["a", 0]
    assert res.data.loc["b", 0] == t1.data.loc["b", 0]


def test_save_experiment_with_mode_w():
    experiment_name = "test"
    a = jb.Artifact({0: {"a": jb.Number(1, "num")}})
    t = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["a"]))
    e = jb.Experiment((a, t))
    jb.save(e, experiment_name, savedir=SAVEDIR, mode="w")


def test_save_experiment_with_mode_a():
    experiment_name = "test"
    a0 = jb.Artifact({0: {"a": jb.Number(1, "num")}})
    t0 = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["a"]))
    e0 = jb.Experiment((a0, t0))
    jb.save(e0, experiment_name, savedir=SAVEDIR, mode="w")

    a1 = jb.Artifact({1: {"a": jb.Number(1, "num")}})
    t1 = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["b"]))
    e1 = jb.Experiment((a1, t1))
    jb.save(e1, experiment_name, savedir=SAVEDIR, mode="a")


def test_load_experiment():
    experiment_name = "test"
    a0 = jb.Artifact({0: {"a": jb.Number(1, "num")}})
    t0 = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["a"]))
    e0 = jb.Experiment((a0, t0))
    jb.save(e0, experiment_name, savedir=SAVEDIR, mode="w")

    a1 = jb.Artifact({1: {"a": jb.Number(1, "num")}})
    t1 = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["b"]))
    e1 = jb.Experiment((a1, t1))
    jb.save(e1, experiment_name, savedir=SAVEDIR, mode="a")

    res = jb.load(experiment_name, savedir=SAVEDIR)

    artifact, table = res.data
    assert artifact.data[0] == a0.data[0]
    assert artifact.data[1] == a1.data[1]
    assert table.data.loc["a", 0] == t0.data.loc["a", 0]
    assert table.data.loc["b", 0] == t1.data.loc["b", 0]
