import datetime as dt
import pandas as pd
from jijbench.id import ID


def test_id():
    test_id = ID(
        benchmark="a",
        experiment="b",
        run="c",
        timestamp=pd.Timestamp(2022, 1, 1),
    )

    assert test_id.benchmark == "a"
    assert test_id.experiment == "b"
    assert test_id.run == "c"
    assert test_id.timestamp == dt.datetime(2022, 1, 1)
