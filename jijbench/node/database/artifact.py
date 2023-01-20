from __future__ import annotations

import typing as tp

from dataclasses import dataclass, field
from jijbench.node import FunctionNode


@dataclass
class Artifact(DataBase):
    data: dict = field(default_factory=dict)

    def append(self, record: Record, **kwargs: tp.Any) -> None:
        self._append(record, ArtifactFactory(), **kwargs)


class ArtifactFactory(FunctionNode["Record", "Artifact"]):
    def __call__(self, inputs: list[Record], name: str | None = None) -> Artifact:
        data = {node.name: node.data.to_dict() for node in inputs}
        return Artifact(data, name=name)

    @property
    def name(self) -> str:
        return "artifact"
