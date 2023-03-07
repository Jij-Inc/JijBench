import numpy as np
import pandas as pd
import jijbench as jb


def test_record_factory():
    factory = jb.functions.RecordFactory()

    inputs = [jb.ID(), jb.Date(), jb.Array(np.arange(5))]
    record = factory(inputs)

    print()
    print(record.data)

    assert isinstance(record, jb.Record)
    assert record.operator is None
    for node in inputs:
        assert node in factory.inputs


def test_table_factory():
    factory = jb.functions.TableFactory()

    data = [jb.ID(), jb.Date(), jb.Array(np.arange(5))]
    r1 = jb.Record(pd.Series(data), "a")
    r2 = jb.Record(pd.Series(data), "b")
    table = factory([r1, r2])

    assert isinstance(table, jb.Table)
    assert table.operator is None
    assert table.data.index[0] == "a"
    assert table.data.index[1] == "b"
    for i, d in enumerate(table.data.loc["a"]):
        assert d == factory.inputs[0].data[i]


def test_artifact_factory():
    factory = jb.functions.ArtifactFactory()

    data = [jb.ID(), jb.Date(), jb.Array(np.arange(5))]
    r1 = jb.Record(pd.Series(data), "a")
    r2 = jb.Record(pd.Series(data), "b")
    artifact = factory([r1, r2])

    assert isinstance(artifact, jb.Artifact)
    assert artifact.operator is None
    assert "a" in artifact.data
    assert "b" in artifact.data
    for i, d in artifact.data["a"].items():
        assert d == factory.inputs[0].data[i]


def test_concat_table():
    concat = jb.functions.Concat()

    factory = jb.functions.TableFactory()
    data = [jb.ID(), jb.Date(), jb.Array(np.arange(5))]
    r1 = jb.Record(pd.Series(data), "a")
    r2 = jb.Record(pd.Series(data), "b")

    t1 = factory([r1])
    t2 = factory([r2])

    table = concat([t1, t2])

    assert isinstance(table, jb.Table)
    assert table.operator is None
    assert table.data.index[0] == "a"
    assert table.data.index[1] == "b"
    for i, d in enumerate(table.data.loc["a"]):
        assert d == factory.inputs[0].data[i]


def test_concat_artifact():
    concat = jb.functions.Concat()

    factory = jb.functions.ArtifactFactory()
    data = [jb.ID(), jb.Date(), jb.Array(np.arange(5))]
    r1 = jb.Record(pd.Series(data), "a")
    r2 = jb.Record(pd.Series(data), "b")

    a1 = factory([r1])
    a2 = factory([r2])
    
    print()
    print(a1.name, a2.name)

    artifact = concat([a1, a2])
    
    print(artifact.data)
    print(artifact.data.keys())

    assert isinstance(artifact, jb.Artifact)
    assert artifact.operator is None
    assert "a" in artifact.data
    assert "b" in artifact.data
    for i, d in artifact.data["a"].items():
        assert d == factory.inputs[0].data[i]


def test_concat():
    concat = jb.functions.Concat()

    factory = jb.functions.TableFactory()
    data = [jb.ID(), jb.Date(), jb.Array(np.arange(5))]
    r1 = jb.Record(pd.Series(data), "a")
    r2 = jb.Record(pd.Series(data), "b")

    t1 = factory([r1])
    t2 = factory([r2])

    e1 = jb.Experiment((a1, t1), name="test1")
    e2 = jb.Experiment((a2, t2), name="test2")

    concat = jb.functions.Concat()
    e = concat([e1, e2])
    print()
    print(e.artifact)
    print(e.table)


# def test_t():
#     from jijbench.typing import DataNodeT
# 
#     print(DataNodeT)
