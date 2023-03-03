from __future__ import annotations

import dill
import pathlib
import typing as tp

from jijbench.consts.path import DEFAULT_RESULT_DIR
from jijbench.mappings.mappings import Artifact, Table
from jijbench.functions.concat import Concat

if tp.TYPE_CHECKING:
    from jijbench.experiment.experiment import Experiment


@tp.overload
def save(
    obj: Experiment,
    *,
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    mode: tp.Literal["w", "a"] = "w",
    index_col: int | list[int] | None = None,
) -> None:
    ...


@tp.overload
def save(
    obj: Artifact,
    *,
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    mode: tp.Literal["w", "a"] = "w",
) -> None:
    ...


@tp.overload
def save(
    obj: Table,
    *,
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    mode: tp.Literal["w", "a"] = "w",
    index_col: int | list[int] | None = None,
) -> None:
    ...


def save(
    obj: Artifact | Experiment | Table,
    *,
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    mode: tp.Literal["w", "a"] = "w",
    index_col: int | list[int] | None = None,
) -> None:
    """Save the given `Artifact`, `Experiment`, or `Table` object.

    Args:
        obj (Artifact | Experiment | Table): The object to be saved.
        savedir (str | pathlib.Path, optional): The directory where the object will be saved. Defaults to DEFAULT_RESULT_DIR.
        mode (Literal[&quot;w&quot;, &quot;a&quot;], optional): The write mode for the file. Must be 'w' or 'a'. Defaults to "w".
        index_col (int | list[int] | None, optional): Index column(s) to set as index while saving the table. Defaults to None. Defaults to None.

    Raises:
        ValueError: If the mode is not 'w' or 'a'.
        FileNotFoundError: If the savedir does not exist.
        IOError: If the object is not dillable.
        TypeError: If the object is not an `Artifact`, `Experiment`, or `Table`.
    """
    from jijbench.experiment.experiment import Experiment

    def is_dillable(obj: tp.Any) -> bool:
        try:
            dill.dumps(obj)
            return True
        except Exception:
            return False

    if mode not in ["a", "w"]:
        raise ValueError("Argument mode must be 'a' or 'b'.")

    savedir = savedir if isinstance(savedir, pathlib.Path) else pathlib.Path(savedir)
    if not savedir.exists():
        raise FileNotFoundError(f"Directory {savedir} is not found.")

    if isinstance(obj, Artifact):
        p = savedir / "artifact.dill"
        concat_a: Concat[Artifact] = Concat()
        if not is_dillable(obj):
            raise IOError(f"Cannot save object: {obj}.")

        if mode == "a":
            if p.exists():
                obj = concat_a(
                    [
                        load(
                            savedir,
                            return_type="Artifact",
                        ),
                        obj,
                    ]
                )

        with open(p, "wb") as f:
            dill.dump(obj, f)

    elif isinstance(obj, Experiment):
        savedir_a = savedir / obj.name / "artifact"
        savedir_t = savedir / obj.name / "table"
        savedir_a.mkdir(parents=True, exist_ok=True)
        savedir_t.mkdir(parents=True, exist_ok=True)
        save(
            obj.data[0],
            savedir=savedir_a,
            mode=mode,
        )
        save(
            obj.data[1],
            savedir=savedir_t,
            mode=mode,
            index_col=index_col,
        )
    elif isinstance(obj, Table):
        p = savedir / "table.csv"
        p_dill = savedir / "table.dill"
        p_meta = savedir / "meta.dill"
        concat_t: Concat[Table] = Concat()
        if mode == "a":
            if p.exists() and p_meta.exists():
                obj = concat_t(
                    [
                        load(
                            savedir,
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
        with open(p_dill, "wb") as f:
            dill.dump(obj, f)
        with open(p_meta, "wb") as f:
            dill.dump(meta, f)
    else:
        raise TypeError(
            f"Cannnot save type {obj.__class__}. Type of obj must be Artifact or Experiment or Table."
        )


@tp.overload
def load(
    name_or_dir: str | pathlib.Path,
    *,
    experiment_names: list[str] | None = None,
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    index_col: int | list[int] | None = None,
) -> Experiment:
    ...


@tp.overload
def load(
    name_or_dir: str | pathlib.Path,
    *,
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    return_type: tp.Literal["Artifact"] = ...,
) -> Artifact:
    ...


@tp.overload
def load(
    name_or_dir: str | pathlib.Path,
    *,
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    return_type: tp.Literal["Table"] = ...,
    index_col: int | list[int] | None = None,
) -> Table:
    ...


def load(
    name_or_dir: str | pathlib.Path,
    *,
    experiment_names: list[str] | None = None,
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    return_type: tp.Literal["Artifact", "Experiment", "Table"] = "Experiment",
    index_col: int | list[int] | None = None,
) -> Experiment | Artifact | Table:
    """Load and return an artifact, experiment, or table from the given directory.

    Args:
        name_or_dir (str | pathlib.Path): Name of the experiment or the directory of the experiment.
        experiment_names (list[str] | None, optional): List of names of experiments to be loaded, if None, all experiments in `savedir` will be loaded. Defaults to None.
        savedir (str | pathlib.Path, optional): Directory of the experiment. Defaults to DEFAULT_RESULT_DIR.
        return_type (tp.Literal[&quot;Artifact&quot;, &quot;Experiment&quot;, &quot;Table&quot;], optional): Type of the returned object. Defaults to "Experiment".
        index_col (int | list[int] | None, optional): The column(s) to set as the index(MultiIndex) of the returned Table.. Defaults to None.

    Raises:
        FileNotFoundError: If `name_or_dir` is not found in the `savedir` directory.
        ValueError: If `return_type` is not one of "Artifact", "Experiment", or "Table".

    Returns:
        Experiment | Artifact | Table: The loaded artifact, experiment, or table.
    """
    from jijbench.experiment.experiment import Experiment

    name_or_dir = (
        name_or_dir
        if isinstance(name_or_dir, pathlib.Path)
        else pathlib.Path(name_or_dir)
    )

    if name_or_dir.exists():
        savedir = name_or_dir
    else:
        savedir /= name_or_dir

    if not savedir.exists():
        raise FileNotFoundError(f"{name_or_dir} is not found.")

    if return_type == "Artifact":
        with open(f"{savedir}/artifact.dill", "rb") as f:
            return dill.load(f)
    elif return_type == "Experiment":
        if experiment_names is None:
            dirs = savedir.iterdir()
        else:
            dirs = [savedir / name for name in experiment_names]
        experiments = []
        for p in dirs:
            if p.is_dir():
                a = load("artifact", savedir=p, return_type="Artifact")
                t = load("table", savedir=p, return_type="Table", index_col=index_col)
                experiments.append(Experiment((a, t), p.name))
        concat: Concat[Experiment] = Concat()
        return concat(experiments)
    elif return_type == "Table":
        with open(f"{savedir}/table.dill", "rb") as f:
            return dill.load(f)
    else:
        raise ValueError("Argument return type must be Artifact, Experiment or Table.")
