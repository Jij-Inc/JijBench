import jijbench as jb
import numpy as np
import pandas as pd

from icecream import ic


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

    name = "r1"
    r1 = factory(inputs1, name=name)
    r2 = factory(inputs2, name="r2")

    r1.append(r2)

    assert isinstance(r1, jb.Record)
    assert r1.name == name
    assert "id2" in r1.data.index
    assert "date2" in r1.data.index
    assert "array2" in r1.data.index
    assert r1.operator is not None
    assert len(r1.operator.inputs) == 2
    for i in ["id2", "date2", "array2"]:
        assert r1.data[i] == r2.data[i]


def test_table_append():
    factory = jb.functions.TableFactory()

    name = "t1"
    data = [jb.ID("id"), jb.Date(), jb.Array(np.arange(5), "array")]
    r1 = jb.Record(pd.Series(data), "a")
    table = factory([r1], name=name)

    r2 = jb.Record(pd.Series(data), "b")
    table.append(r2)

    assert isinstance(table, jb.Table)
    assert table.name == name
    assert table.data.index[0] == "a"
    assert table.data.index[1] == "b"
    assert table.operator is not None
    assert len(table.operator.inputs) == 2
    for i, d in enumerate(table.data.loc["a"]):
        assert d == table.operator.inputs[0].data.loc["a", i]


def test_artifact_append():
    factory = jb.functions.ArtifactFactory()

    name = "a1"
    data = [jb.ID("id"), jb.Date(), jb.Array(np.arange(5), "array")]
    r1 = jb.Record(pd.Series(data), "a")
    artifact = factory([r1], name=name)

    r2 = jb.Record(pd.Series(data), "b")
    artifact.append(r2)

    assert isinstance(artifact, jb.Artifact)
    assert artifact.name == name
    assert "a" in artifact.data
    assert "b" in artifact.data
    assert artifact.operator is not None
    assert len(artifact.operator.inputs) == 2
    for i, d in artifact.data["a"].items():
        assert d == artifact.operator.inputs[0].data["a"][i]


def test_experiment_append():
    e = jb.Experiment(name="test")

    for i in range(3):
        data = [
            jb.ID(name="id"),
            jb.Date(),
            jb.Number(i, "num"),
            jb.Array(np.arange(5), "array"),
        ]
        record = jb.functions.RecordFactory()(data, name=i)
        e.append(record)
    
    ic()
    ic(e.operator)
    ic(len(e.operator.inputs))
    ic(e.data[1].data.applymap(id))
    ic(e.data[1].data.applymap(id))
    ic(e.operator.inputs[0].data[1].data.applymap(id))

    assert isinstance(e, jb.Experiment)
    assert e.operator is not None
    assert e.data[0].operator is not None
    assert e.data[1].operator is not None
    assert len(e.operator.inputs) == 2

    for i in range(3):
        assert i in e.artifact
        assert e.artifact[i]["num"] == i
        assert e.table.loc[i, "num"] == i


def test_ref():
    factory = jb.functions.RecordFactory()
    inputs1 = [
        jb.ID(name="id1"),
        jb.Date(name="date1"),
        jb.Array(np.arange(5), name="array1"),
    ]

    r1 = factory(inputs1)

    factory = jb.functions.TableFactory()
    table = r1.apply(factory)
    ic()
    ic(table.data.applymap(id))
    ic(table.operator.inputs[0].data.apply(id))
    ic(list(map(id, inputs1)))
    ic(table.operator.inputs[0].operator)
    # ic(":::::::")
    # ic(e.operator)
    # ic(len(e.operator.inputs))
    # ic(e.data[1])
    # ic(type(e.data[1]))
    # ic(e.data[1].operator)
    # ic(e.data[1].operator.inputs[0].data.applymap(id))
    # ic(e.data[1].operator.inputs[1].data.applymap(id))
    # TODO add test
    # ic(e.data[1].operator.inputs[2].data)
    # ic(e.data[1].operator.inputs[3].data)
    # ic(e.data[1].operator.inputs[4].data)
    # ic(e.data[1].operator.inputs[5].data)
    # ic(e.data[1].data.applymap(id))
    # ic(list(map(id, e.data[1].operator.inputs[1].operator.inputs[0].operator.inputs)))
    # ic(list(map(id, e.data[0].operator.inputs[1].operator.inputs[0].operator.inputs)))
    # ic("=========")
    # ic(
    #     e.operator.inputs[0]
    #     .operator.inputs[1]
    #     .data[0]
    #     .operator.inputs[0]
    #     .operator.inputs
    # )
    # print(e.operator.inputs[0].operator.inputs[0])
    # print(e.operator.inputs[0].operator.inputs[0].operator)
    # print(e.operator.inputs[0].operator.inputs[0].operator.inputs)
