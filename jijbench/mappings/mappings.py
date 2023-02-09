from __future__ import annotations

import abc
import pandas as pd
import typing as tp

from dataclasses import dataclass, field
from jijbench.functions.concat import Concat
from jijbench.functions.factory import ArtifactFactory, TableFactory
from jijbench.node.base import DataNode
from jijbench.typing import T, ArtifactDataType


@dataclass
class Mapping(DataNode[T]):
    @abc.abstractmethod
    def append(self, record: Record) -> None:
        pass

    @abc.abstractmethod
    def view(self) -> T:
        pass


@dataclass
class Record(Mapping[pd.Series]):
    data: pd.Series = field(default_factory=lambda: pd.Series(dtype="object"))
    name: tp.Hashable = None

    @classmethod
    def validate_data(cls, data: pd.Series) -> pd.Series:
        data = cls._validate_dtype(data, (pd.Series,))
        if data.empty:
            return data
        else:
            if data.apply(lambda x: isinstance(x, DataNode)).all():
                return data
            else:
                raise TypeError(
                    f"All elements of {data.__class__.__name__} must be type DataNode."
                )

    @property
    def index(self) -> pd.Index:
        return self.data.index

    @index.setter
    def index(self, index: pd.Index) -> None:
        self.data.index = index

    def append(self, record: Record) -> None:
        concat: Concat[Record] = Concat()
        node = self.apply(concat, [record], name=self.name)
        self.__init__(**node.__dict__)

    def view(self) -> pd.Series:
        return self.data.apply(lambda x: x.data)


@dataclass
class Artifact(Mapping[ArtifactDataType]):
    data: ArtifactDataType = field(default_factory=dict)
    name: tp.Hashable = None

    @classmethod
    def validate_data(cls, data: ArtifactDataType) -> ArtifactDataType:
        if data:
            data = cls._validate_dtype(data, (dict,))
            values = []
            for v in data.values():
                if isinstance(v, dict):
                    values += list(v.values())
                else:
                    raise TypeError(
                        f"Type of attibute data is {ArtifactDataType}. Input data is invaid."
                    )
            if all(map(lambda x: isinstance(x, DataNode), values)):
                return data
            else:
                raise TypeError(
                    f"Type of attibute data is {ArtifactDataType}. Input data is invaid."
                )
        else:
            return data

    def keys(self) -> tuple[tp.Hashable, ...]:
        return tuple(self.data.keys())

    def values(self) -> tuple[dict[tp.Hashable, DataNode], ...]:
        return tuple(self.data.values())

    def items(
        self,
    ) -> tuple[tuple[tp.Hashable, dict[tp.Hashable, DataNode]], ...]:
        return tuple(self.data.items())

    def append(self, record: Record) -> None:
        concat: Concat[Artifact] = Concat()
        other = ArtifactFactory()([record])
        node = self.apply(concat, [other], name=self.name)
        self.__init__(**node.__dict__)

    def view(self) -> ArtifactDataType:
        return {
            k: {name: node.data for name, node in v.items()}
            for k, v in self.data.items()
        }


@dataclass
class Table(Mapping[pd.DataFrame]):
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    name: tp.Hashable = None

    @classmethod
    def validate_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        data = cls._validate_dtype(data, (pd.DataFrame,))
        if data.empty:
            return data
        else:
            if data.applymap(lambda x: isinstance(x, DataNode)).values.all():
                return data
            else:
                raise TypeError(
                    f"All elements of {data.__class__.__name__} must be type DataNode."
                )

    @property
    def index(self) -> pd.Index:
        return self.data.index

    @index.setter
    def index(self, index: pd.Index) -> None:
        self.data.index = index

    @property
    def columns(self) -> pd.Index:
        return self.data.columns

    @columns.setter
    def columns(self, columns: pd.Index) -> None:
        self.data.columns = columns

    def append(self, record: Record) -> None:
        concat: Concat[Table] = Concat()
        other = TableFactory()([record])
        node = self.apply(concat, [other], name=self.name, axis=0)
        self.__init__(**node.__dict__)

    def view(self) -> pd.DataFrame:
        if self.data.empty:
            return self.data
        else:
            is_tuple_index = all([isinstance(i, tuple) for i in self.data.index])
            if is_tuple_index:
                names = (
                    self.data.index.names if len(self.data.index.names) >= 2 else None
                )
                index = pd.MultiIndex.from_tuples(self.data.index, names=names)
                # TODO 代入しない
                self.data.index = index
            return self.data.applymap(lambda x: x.data)
