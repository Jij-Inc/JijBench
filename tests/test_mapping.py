import jijbench as jb
import numpy as np
import pandas as pd


def test_table():
    pass


def test_record_append():
    factory = jb.functions.RecordFactory()

    inputs1 = [
        jb.ID(name="id1"),
        jb.Date(name="date1"),
        jb.Array(np.arange(5), name="array1"),
    ]
    inputs2 = [
        jb.ID(name="id2"),
        jb.Date(name="date2"),
        jb.Array(np.arange(5), name="array2"),
    ]

    r1 = factory(inputs1)
    r2 = factory(inputs2)

    r1.append(r2)

    assert isinstance(r1, jb.Record)
    assert "id2" in r1.data.index
    assert "date2" in r1.data.index
    assert "array2" in r1.data.index

    for i in ["id2", "date2", "array2"]:
        assert r1.data[i] == r2.data[i]


def test_table_append():
    factory = jb.functions.TableFactory()

    data = [jb.ID(), jb.Date(), jb.Array(np.arange(5))]
    r1 = jb.Record(pd.Series(data), "a")
    table = factory([r1])

    r2 = jb.Record(pd.Series(data), "b")
    table.append(r2)

    assert isinstance(table, jb.Table)
    assert table.data.index[0] == "a"
    assert table.data.index[1] == "b"
    assert table.operator is not None
    for i, d in enumerate(table.data.loc["a"]):
        assert d == table.operator.inputs[0].data.loc["a", i]


def test_artifact_append():
    factory = jb.functions.ArtifactFactory()

    data = [jb.ID(), jb.Date(), jb.Array(np.arange(5))]
    r1 = jb.Record(pd.Series(data), "a")
    artifact = factory([r1])

    r2 = jb.Record(pd.Series(data), "b")
    artifact.append(r2)

    assert isinstance(artifact, jb.Artifact)
    assert "a" in artifact.data
    assert "b" in artifact.data
    assert artifact.operator is not None
    for i, d in artifact.data["a"].items():
        assert d == artifact.operator.inputs[0].data["a"][i]


def test_experiment_append():
    pass
