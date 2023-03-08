import dill
import jijbench as jb
import pandas as pd
import pathlib
import pytest
import shutil


SAVEDIR = "./jb_results"


@pytest.fixture(scope="function", autouse=True)
def pre_post_process():
    # preprocess
    p = pathlib.Path(SAVEDIR)
    p.mkdir(exist_ok=True)
    yield
    # postprocess
    p = pathlib.Path(SAVEDIR)
    if p.exists():
       shutil.rmtree(p)


def test_save_artifact_with_mode_w():
    a = jb.Artifact({0: {"a": jb.Number(1, "num")}})
    jb.save(a, savedir=SAVEDIR)

    with open(f"{SAVEDIR}/artifact.dill", "rb") as f:
        res = dill.load(f)

    assert res == a


def test_save_artifact_with_mode_a():
    a0 = jb.Artifact({0: {"a": jb.Number(1, "num")}})
    jb.save(a0, savedir=SAVEDIR)

    a1 = jb.Artifact({1: {"a": jb.Number(1, "num")}})
    jb.save(a1, savedir=SAVEDIR, mode="a")

    with open(f"{SAVEDIR}/artifact.dill", "rb") as f:
        res = dill.load(f)

    assert res.data[0] == a0.data[0]
    assert res.data[1] == a1.data[1]


def test_load_artifact():
    a0 = jb.Artifact({0: {"a": jb.Number(1, "num")}})
    jb.save(a0, savedir=SAVEDIR)

    a1 = jb.Artifact({1: {"a": jb.Number(1, "num")}})
    jb.save(a1, savedir=SAVEDIR, mode="a")

    res = jb.load(SAVEDIR, return_type="Artifact")
    assert res.data[0] == a0.data[0]
    assert res.data[1] == a1.data[1]


def test_save_table_with_mode_w():
    t = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["a"]))
    jb.save(t, savedir=SAVEDIR)

    res = pd.read_csv(f"{SAVEDIR}/table.csv", index_col=0)
    with open(f"{SAVEDIR}/meta.dill", "rb") as f:
        meta = dill.load(f)

    assert res.loc["a", "0"] == 1
    assert meta["dtype"][0] == jb.Number


def test_save_table_with_mode_a():
    t0 = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["a"]))
    jb.save(t0, savedir=SAVEDIR)

    t1 = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["b"]))
    jb.save(t1, savedir=SAVEDIR, mode="a")

    res = pd.read_csv(f"{SAVEDIR}/table.csv", index_col=0)
    with open(f"{SAVEDIR}/meta.dill", "rb") as f:
        meta = dill.load(f)

    assert res.loc["a", "0"] == 1
    assert res.loc["b", "0"] == 1
    assert meta["dtype"][0] == jb.Number


def test_load_table():
    t0 = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["a"]))
    jb.save(t0, savedir=SAVEDIR)

    t1 = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["b"]))
    jb.save(t1, savedir=SAVEDIR, mode="a")

    res = jb.load(SAVEDIR, return_type="Table")
    assert res.data.loc["a", 0] == t0.data.loc["a", 0]
    assert res.data.loc["b", 0] == t1.data.loc["b", 0]


def test_save_experiment_with_mode_w():
    experiment_name = "0"
    a = jb.Artifact({0: {"a": jb.Number(1, "num")}})
    t = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["a"]))
    e = jb.Experiment((a, t), name=experiment_name)
    jb.save(e, savedir=SAVEDIR, mode="w")


def test_save_experiment_with_mode_a():
    experiment_name = "0"
    a0 = jb.Artifact({0: {"a": jb.Number(1, "num")}})
    t0 = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["a"]))
    e0 = jb.Experiment((a0, t0), name=experiment_name)
    jb.save(e0, savedir=SAVEDIR, mode="w")

    a1 = jb.Artifact({1: {"a": jb.Number(1, "num")}})
    t1 = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["b"]))
    e1 = jb.Experiment((a1, t1), name=experiment_name)
    jb.save(e1, savedir=SAVEDIR, mode="a")


def test_load_experiment():
    experiment_name = "0"
    a0 = jb.Artifact({0: {"a": jb.Number(1, "num")}})
    t0 = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["a"]))
    e0 = jb.Experiment((a0, t0), name=experiment_name)
    jb.save(e0, savedir=SAVEDIR, mode="w")

    a1 = jb.Artifact({1: {"a": jb.Number(1, "num")}})
    t1 = jb.Table(pd.DataFrame([[jb.Number(1, "num")]], index=["b"]))
    e1 = jb.Experiment((a1, t1), name=experiment_name)
    jb.save(e1, savedir=SAVEDIR, mode="a")

    res = jb.load(SAVEDIR, experiment_names=[experiment_name])

    artifact, table = res.data
    assert artifact.data[0] == a0.data[0]
    assert artifact.data[1] == a1.data[1]
    assert table.data.loc["a", 0] == t0.data.loc["a", 0]
    assert table.data.loc["b", 0] == t1.data.loc["b", 0]


def test_load_benchmark_results():
    def func(x):
        return x

    benchmark_name = "test"
    bench = jb.Benchmark({"x": [1, 2]}, solver=func, name=benchmark_name)
    res = bench(savedir=SAVEDIR)

    res_l = jb.load(benchmark_name, savedir=SAVEDIR)
