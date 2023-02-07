from __future__ import annotations

import dill
import pathlib
import pandas as pd
import typing as tp

from jijbench.consts.path import DEFAULT_RESULT_DIR
from jijbench.experiment.experiment import Experiment
from jijbench.mappings.mappings import Artifact, Table
from jijbench.functions.concat import Concat


@tp.overload
def save(
    obj: Artifact,
    experiment_name: str,
    *,
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    mode: tp.Literal["w", "a"] = "w",
) -> None:
    ...


@tp.overload
def save(
    obj: Experiment,
    experiment_name: str,
    *,
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    mode: tp.Literal["w", "a"] = "w",
) -> None:
    ...


@tp.overload
def save(
    obj: Table,
    experiment_name: str,
    *,
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    mode: tp.Literal["w", "a"] = "w",
    index_col: int | list[int] | None = 0,
) -> None:
    ...


def save(
    obj: Artifact | Experiment | Table,
    experiment_name: str,
    *,
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    mode: tp.Literal["w", "a"] = "w",
    index_col: int | list[int] | None = 0,
) -> None:
    def is_dillable(obj: tp.Any) -> bool:
        try:
            dill.dumps(obj)
            return True
        except Exception:
            return False

    if mode not in ["a", "w"]:
        raise ValueError("Argument mode must be 'a' or 'b'.")

    base_savedir = (
        savedir if isinstance(savedir, pathlib.Path) else pathlib.Path(savedir)
    )
    if not base_savedir.exists():
        raise FileNotFoundError(f"Directory {savedir} is not found.")

    savedir = base_savedir / experiment_name
    savedir.mkdir(exist_ok=True)

    if isinstance(obj, Artifact):
        p = savedir / "artifact.dill"
        concat_a: Concat[Artifact] = Concat()
        if not is_dillable(obj):
            raise IOError(f"Cannot save object: {obj}.")

        if mode == "a":
            if p.exists():
                obj = concat_a(
                    [
                        load(experiment_name, savedir=base_savedir, return_type="Artifact"),
                        obj,
                    ]
                )

        with open(p, "wb") as f:
            dill.dump(obj, f)

    elif isinstance(obj, Experiment):
        savedir_a = savedir / "artifact"
        savedir_a.mkdir(exist_ok=True)
        savedir_t = savedir / "table"
        savedir_t.mkdir(exist_ok=True)
        save(obj.data[0], experiment_name, savedir=savedir_a, mode=mode)
        save(
            obj.data[1],
            experiment_name,
            savedir=savedir_t,
            mode=mode,
            index_col=index_col,
        )
    elif isinstance(obj, Table):
        p = savedir / "table.csv"
        p_meta = savedir / "meta.dill"
        concat_t: Concat[Table] = Concat()
        if mode == "a":
            if p.exists() and p_meta.exists():
                obj = concat_t(
                    [
                        load(
                            experiment_name,
                            savedir=base_savedir,
                            return_type="Table",
                            index_col=index_col,
                        ),
                        obj,
                    ]
                )
        obj.view().to_csv(p)
        meta = {
            "dtype": obj.data.iloc[0].apply(lambda x: x.__class__).to_dict(),
            "name": obj.data.applymap(lambda x: x.name).to_dict(),
            "index": obj.data.index,
            "columns": obj.data.columns,
        }
        with open(p_meta, "wb") as f:
            dill.dump(meta, f)
    else:
        raise TypeError(
            f"Cannnot save type {obj.__class__}. Type of obj must be Artifact or Experiment or Table."
        )


@tp.overload
def load(
    experiment_name: str,
    *,
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
) -> Experiment:
    ...


@tp.overload
def load(
    experiment_name: str,
    *,
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    return_type: tp.Literal["Artifact"] = ...,
) -> Artifact:
    ...


@tp.overload
def load(
    experiment_name: str,
    *,
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    return_type: tp.Literal["Table"] = ...,
    index_col: int | list[int] | None = 0,
) -> Table:
    ...


def load(
    experiment_name: str,
    *,
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    return_type: tp.Literal["Artifact", "Experiment", "Table"] = "Experiment",
    index_col: int | list[int] | None = 0,
) -> Experiment | Artifact | Table:
    savedir = savedir if isinstance(savedir, pathlib.Path) else pathlib.Path(savedir)
    savedir /= experiment_name

    if not savedir.exists():
        raise FileNotFoundError(f"Directory {savedir} is not found.")

    if return_type == "Artifact":
        with open(f"{savedir}/artifact.dill", "rb") as f:
            return dill.load(f)
    elif return_type == "Experiment":
        savedir_a = savedir / "artifact"
        savedir_t = savedir / "table"
        a = load(experiment_name, savedir=savedir_a, return_type="Artifact")
        b = load(experiment_name, savedir=savedir_t, return_type="Table")
        return Experiment((a, b))
    elif return_type == "Table":
        p = savedir / "table.csv"
        p_meta = savedir / "meta.dill"
        data = pd.read_csv(p, index_col=index_col)
        with open(p_meta, "rb") as f:
            meta = dill.load(f)
        data.index = meta["index"]
        data.columns = meta["columns"]
        data = data.apply(
            lambda x: [
                meta["dtype"][x.name](data, meta["name"][x.name][index])
                for index, data in zip(x.index, x)
            ]
        )
        return Table(data)
    else:
        raise ValueError
