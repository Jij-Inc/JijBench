import datetime
import numpy as np
import pandas as pd
import pytest
import jijbench as jb


def test_construct_array():
    array = jb.Array(np.array([1, 2]), name="sample")

    assert array.data[0] == 1


def test_construct_date():
    date = jb.Date()

    assert isinstance(date.data, pd.Timestamp)

    date = jb.Date("2023-01-01")

    assert date.data == pd.Timestamp(2023, 1, 1)

    date = jb.Date(datetime.datetime(2023, 1, 1))

    assert date.data == pd.Timestamp(2023, 1, 1)


def test_construct_number():
    x = jb.Number(1, "x")

    assert x.data == 1

    x = jb.Number(1.0, "x")

    assert x.data == 1.0


def test_construct_string():
    s = jb.String("a", "s")

    assert s.data == "a"


def test_construct_callable():
    def f():
        pass

    c = jb.Callable(f, "f")

    assert c.data == f


def test_array_invalid_data():
    with pytest.raises(TypeError):
        jb.Array([1, 2], name="sample")


def test_date_invalid_data():
    with pytest.raises(TypeError):
        jb.Date(1)

    with pytest.raises(ValueError):
        jb.Date("a")


def test_number_invalid_data():
    with pytest.raises(TypeError):
        jb.Number("1", "x")


def test_string_invalid_data():
    with pytest.raises(TypeError):
        jb.String(1, "x")


def test_callable_invalid_data():
    with pytest.raises(TypeError):
        jb.Callable(1, "f")


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
