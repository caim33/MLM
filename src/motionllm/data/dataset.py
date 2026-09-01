"""Identity-preserving in-memory dataset for canonical samples."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence

from motionllm.contracts import Sample, SampleContractError

from .samples import read_samples_jsonl


class SampleDataset(Sequence[Sample]):
    """An immutable sequence that never retries with a different row."""

    def __init__(self, samples: Iterable[Sample]) -> None:
        materialized = tuple(samples)
        seen: set[str] = set()
        for index, sample in enumerate(materialized):
            if not isinstance(sample, Sample):
                raise SampleContractError(f"dataset item {index} must be a Sample")
            if sample.sample_id in seen:
                raise SampleContractError(f"duplicate sample_id: {sample.sample_id}")
            seen.add(sample.sample_id)
        self._samples = materialized

    @classmethod
    def from_jsonl(
        cls,
        source: str | os.PathLike[str],
        *,
        media_root: str | os.PathLike[str] | None = None,
        check_media_exists: bool = True,
    ) -> "SampleDataset":
        return cls(
            read_samples_jsonl(
                source,
                media_root=media_root,
                check_media_exists=check_media_exists,
            )
        )

    @property
    def samples(self) -> tuple[Sample, ...]:
        return self._samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int | slice) -> Sample | tuple[Sample, ...]:
        return self._samples[index]

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)
