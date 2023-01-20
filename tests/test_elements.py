import numpy as np
import pandas as pd
import jijbench as jb


def test_array_min():
    array = jb.Array(np.arange(5), name="sample")
    res = array.min()

    assert res.data == 0.0
    assert isinstance(res.operator, jb.functions.Min)
    assert array in res.operator.inputs


def test_array_max():
    array = jb.Array(np.arange(5), name="sample")
    res = array.max()

    assert res.data == 4.0
    assert isinstance(res.operator, jb.functions.Max)
    assert array in res.operator.inputs


def test_array_mean():
    array = jb.Array(np.array([1.0, 1.0]), name="sample")
    res = array.mean()

    assert res.data == 1.0
    assert isinstance(res.operator, jb.functions.Mean)
    assert array in res.operator.inputs


def test_array_std():
    array = jb.Array(np.array([1.0, 1.0]), name="sample")
    res = array.std()

    assert res.data == 0.0
    assert isinstance(res.operator, jb.functions.Std)
    assert array in res.operator.inputs
    

def test_date():
    date = jb.Date()
    
    assert isinstance(date.data, pd.Timestamp)
    
    date = jb.Date("2023-01-01")
    
    assert date.data == pd.Timestamp(2023, 1, 1)
