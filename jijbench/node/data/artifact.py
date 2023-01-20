from __future__ import annotations

import typing as tp

from dataclasses import dataclass, field
import jijbench.node.data.database as _database
import jijbench.node.functions.factory as _factory

if tp.TYPE_CHECKING:
    from jijbench.node.data import Record


@dataclass
class Artifact(_database.DataBase):
    data: dict = field(default_factory=dict)

    def append(self, record: Record, **kwargs: tp.Any) -> None:
        self._append(record, _factory.ArtifactFactory(), **kwargs)
