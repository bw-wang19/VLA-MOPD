"""Dataset loading and deterministic domain mixing for MOPD."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
import random

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from datasets import load_from_disk

from .config import DataConfig


_JSON_SUFFIXES = {".json", ".jsonl", ".ndjson"}
_PARQUET_SUFFIXES = {".parquet", ".pq"}


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _domain_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("domain names must be non-empty strings")
    return value


def _dataset_from_saved_object(value: object, path: Path) -> Dataset:
    if isinstance(value, Dataset):
        return value
    if isinstance(value, DatasetDict):
        if "train" in value:
            return value["train"]
        if len(value) == 1:
            return next(iter(value.values()))
        raise ValueError(
            f"saved DatasetDict at {path} has no 'train' split and is ambiguous: "
            f"{sorted(value)}"
        )
    raise TypeError(f"unsupported datasets object loaded from {path}: {type(value)!r}")


def _read_dataset(path: Path, num_workers: int) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"training data does not exist: {path}")

    if path.is_dir():
        try:
            saved = load_from_disk(str(path))
        except Exception as error:
            raise ValueError(
                f"directory is not a datasets save_to_disk artifact: {path}"
            ) from error
        return _dataset_from_saved_object(saved, path)

    suffix = path.suffix.lower()
    loader = None
    if suffix in _JSON_SUFFIXES:
        loader = "json"
    elif suffix in _PARQUET_SUFFIXES:
        loader = "parquet"
    if loader is None:
        supported = sorted(_JSON_SUFFIXES | _PARQUET_SUFFIXES)
        raise ValueError(
            f"unsupported training data format for {path}; expected one of "
            f"{supported} or a datasets save_to_disk directory"
        )

    kwargs: dict[str, object] = {
        "data_files": str(path),
        "split": "train",
    }
    if num_workers > 1:
        kwargs["num_proc"] = num_workers
    loaded = load_dataset(loader, **kwargs)
    if not isinstance(loaded, Dataset):
        raise TypeError(f"expected Dataset from {path}, got {type(loaded)!r}")
    return loaded


def _validate_prompt_records(dataset: Dataset, domain: str, path: Path) -> None:
    """Reject records that cannot be rendered into a student prompt."""

    for index in range(len(dataset)):
        record = dataset[index]
        has_prompt = isinstance(record.get("prompt"), str)
        has_messages = isinstance(record.get("messages"), list)
        if not (has_prompt or has_messages):
            raise ValueError(
                f"record {index} in domain {domain!r} ({path}) must contain "
                "either a string 'prompt' or a list 'messages'"
            )


def load_domain_dataset(
    path: str | Path,
    domain: str,
    *,
    num_workers: int = 0,
) -> Dataset:
    """Load one JSON/JSONL/Parquet/saved dataset and attach its domain.

    The domain supplied by the experiment configuration is authoritative.  An
    existing ``domain`` column is replaced so stale labels cannot silently
    route samples to the wrong teacher.
    """

    domain = _domain_name(domain)
    _nonnegative_int(num_workers, "num_workers")
    source_path = Path(path).expanduser()
    dataset = _read_dataset(source_path, num_workers)
    if len(dataset) == 0:
        raise ValueError(f"domain {domain!r} has no training records: {source_path}")
    _validate_prompt_records(dataset, domain, source_path)

    if "domain" in dataset.column_names:
        dataset = dataset.remove_columns("domain")
    return dataset.add_column("domain", [domain] * len(dataset))


def load_domain_datasets(
    train_files: Mapping[str, str | Path],
    *,
    num_workers: int = 0,
) -> dict[str, Dataset]:
    """Load each configured domain in stable domain-name order."""

    if not isinstance(train_files, Mapping) or not train_files:
        raise ValueError("train_files must be a non-empty domain-to-path mapping")
    _nonnegative_int(num_workers, "num_workers")

    normalized: dict[str, str | Path] = {}
    for domain, path in train_files.items():
        normalized[_domain_name(domain)] = path
    return {
        domain: load_domain_dataset(
            normalized[domain], domain, num_workers=num_workers
        )
        for domain in sorted(normalized)
    }


def _normalize_ratios(
    domains: list[str], domain_ratios: Mapping[str, float]
) -> dict[str, float]:
    if not isinstance(domain_ratios, Mapping):
        raise ValueError("domain_ratios must be a mapping")
    if set(domain_ratios) != set(domains):
        raise ValueError(
            "domain_ratios keys must match domain_datasets exactly: "
            f"datasets={sorted(domains)}, ratios={sorted(domain_ratios)}"
        )

    ratios: dict[str, float] = {}
    for domain in domains:
        value = domain_ratios[domain]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"ratio for domain {domain!r} must be finite and positive")
        ratios[domain] = float(value)
    total = math.fsum(ratios.values())
    return {domain: ratios[domain] / total for domain in domains}


def _largest_remainder_quotas(
    ratios: Mapping[str, float], total: int
) -> dict[str, int]:
    """Allocate exactly ``total`` slots with deterministic tie-breaking."""

    exact = {domain: ratio * total for domain, ratio in ratios.items()}
    quotas = {domain: math.floor(value) for domain, value in exact.items()}
    remaining = total - sum(quotas.values())
    priority = sorted(
        ratios,
        key=lambda domain: (-(exact[domain] - quotas[domain]), domain),
    )
    for domain in priority[:remaining]:
        quotas[domain] += 1
    return quotas


def _seed_for_domain(seed: int, domain: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}\0{domain}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def _sample_indices(length: int, count: int, seed: int) -> list[int]:
    """Sample shuffled epochs, repeating only when a quota exceeds a dataset."""

    generator = random.Random(seed)
    selected: list[int] = []
    while len(selected) < count:
        epoch = list(range(length))
        generator.shuffle(epoch)
        selected.extend(epoch[: count - len(selected)])
    return selected


def interleave_by_domain_ratio(
    domain_datasets: Mapping[str, Dataset],
    domain_ratios: Mapping[str, float],
    *,
    seed: int,
    num_samples: int | None = None,
) -> Dataset:
    """Return an exactly ratio-allocated, deterministically interleaved dataset.

    ``num_samples`` is the exact output length.  When omitted, the sum of the
    source lengths is used.  A domain is sampled without replacement within an
    epoch and reshuffled before any required repeat.  Largest-remainder quota
    allocation makes integer counts reproducible and as close as possible to
    the requested ratios.
    """

    if not isinstance(domain_datasets, Mapping) or not domain_datasets:
        raise ValueError("domain_datasets must be a non-empty mapping")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    domains = sorted(_domain_name(domain) for domain in domain_datasets)
    datasets_by_domain: dict[str, Dataset] = {}
    for domain in domains:
        dataset = domain_datasets[domain]
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"domain_datasets[{domain!r}] must be a datasets.Dataset"
            )
        if len(dataset) == 0:
            raise ValueError(f"domain dataset {domain!r} is empty")
        datasets_by_domain[domain] = dataset

    ratios = _normalize_ratios(domains, domain_ratios)
    if num_samples is None:
        output_length = sum(len(dataset) for dataset in datasets_by_domain.values())
    else:
        output_length = _positive_int(num_samples, "num_samples")

    quotas = _largest_remainder_quotas(ratios, output_length)
    schedule = [
        domain for domain in domains for _ in range(quotas[domain])
    ]
    random.Random(seed).shuffle(schedule)
    positions: dict[str, list[int]] = {domain: [] for domain in domains}
    for position, domain in enumerate(schedule):
        positions[domain].append(position)

    existing_columns = {
        name
        for dataset in datasets_by_domain.values()
        for name in dataset.column_names
    }
    order_column = "__mopd_interleave_order__"
    while order_column in existing_columns:
        order_column = f"_{order_column}"

    pieces: list[Dataset] = []
    for domain in domains:
        count = quotas[domain]
        if count == 0:
            continue
        source = datasets_by_domain[domain]
        indices = _sample_indices(
            len(source), count, _seed_for_domain(seed, domain)
        )
        piece = source.select(indices).add_column(order_column, positions[domain])
        pieces.append(piece)

    mixed = concatenate_datasets(pieces)
    mixed = mixed.sort(order_column)
    return mixed.remove_columns(order_column)


def load_training_dataset(config: DataConfig, *, seed: int) -> Dataset:
    """Load and mix all domains described by a validated ``DataConfig``."""

    domain_datasets = load_domain_datasets(
        config.train_files, num_workers=config.num_workers
    )
    available = sum(len(dataset) for dataset in domain_datasets.values())
    sample_count = (
        available
        if config.max_train_samples is None
        else min(available, config.max_train_samples)
    )
    return interleave_by_domain_ratio(
        domain_datasets,
        config.domain_ratios,
        seed=seed,
        num_samples=sample_count,
    )


def build_training_dataset(config: DataConfig, *, seed: int) -> Dataset:
    """Descriptive alias for :func:`load_training_dataset`."""

    return load_training_dataset(config, seed=seed)


__all__ = [
    "build_training_dataset",
    "interleave_by_domain_ratio",
    "load_domain_dataset",
    "load_domain_datasets",
    "load_training_dataset",
]
