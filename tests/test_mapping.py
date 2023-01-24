import jijbench as jb
import numpy as np
import pandas as pd


def test_table():
    pass


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
        
def 
