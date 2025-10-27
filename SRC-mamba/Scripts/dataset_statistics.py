"""Summarise SRC-compatible datasets into a compact statistics table."""
from __future__ import annotations

import argparse
import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np


@dataclass
class DatasetStats:
    name: str
    learners: int
    exercises: int
    concepts: int
    interactions: int
    avg_concepts_per_exercise: float
    avg_exercises_per_concept: float
    avg_interaction_length: float


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be interpreted."""


def _parse_dataset_argument(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Expected NAME=PATH for --dataset arguments (e.g. assist09=data/assist09)"
        )
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Dataset name must be non-empty")
    resolved = Path(path.strip()).expanduser().resolve()
    return name, resolved


def _coerce_path(value: Path | str) -> Path:
    """Normalise user-supplied paths to absolute :class:`Path` objects."""

    if isinstance(value, Path):
        return value
    return Path(value).expanduser().resolve()


def _resolve_npz_path(name: str, raw_path: Path | str) -> Path:
    path = _coerce_path(raw_path)
    if path.is_file() and path.suffix == ".npz":
        return path
    if path.is_dir():
        candidate = path / f"{name}.npz"
        if candidate.exists():
            return candidate
        npz_files = sorted(path.glob("*.npz"))
        if len(npz_files) == 1:
            return npz_files[0]
        available = ", ".join(p.name for p in npz_files) or "<none>"
        raise DatasetError(
            f"Unable to identify NPZ file for dataset '{name}' in {path}. "
            f"Available files: {available}"
        )
    raise DatasetError(f"Path '{path}' is neither an NPZ file nor a directory")


def _iter_sequences(array: np.ndarray) -> Iterator[np.ndarray]:
    for seq in array:
        if isinstance(seq, np.ndarray):
            yield seq.astype(np.int64, copy=False)
        else:
            yield np.asarray(seq, dtype=np.int64)


def _iter_pairs(
    question_sequences: Sequence[np.ndarray],
    skill_sequences: Sequence[np.ndarray],
) -> Iterator[Tuple[int, int]]:
    for q_seq, s_seq in zip(question_sequences, skill_sequences):
        q_arr = np.asarray(q_seq, dtype=np.int64)
        s_arr = np.asarray(s_seq, dtype=np.int64)
        length = min(len(q_arr), len(s_arr))
        if not length:
            continue
        for question, concept in zip(q_arr[:length], s_arr[:length]):
            yield int(question), int(concept)


def _compute_dataset_statistics(name: str, npz_path: Path) -> DatasetStats:
    with np.load(npz_path, allow_pickle=True) as data:
        try:
            skill_sequences = list(_iter_sequences(data["skill"]))
        except KeyError as exc:
            raise DatasetError(f"{npz_path} is missing the 'skill' array") from exc

        learners = len(skill_sequences)
        if not learners:
            raise DatasetError(f"Dataset '{name}' is empty")

        real_len = data.get("real_len")
        if real_len is None:
            raise DatasetError(f"{npz_path} is missing the 'real_len' array")
        real_len = np.asarray(real_len, dtype=np.int64)
        interactions = int(real_len.sum())
        avg_interaction_length = float(real_len.mean()) if learners else 0.0

        concept_ids = data.get("concept_ids")
        if concept_ids is None:
            # Fall back to concepts discovered in the skill sequences.
            concept_set = {int(token) for seq in skill_sequences for token in seq if token >= 0}
            concepts = len(concept_set)
        else:
            concept_ids = np.asarray(concept_ids, dtype=np.int64)
            concepts = int(len(concept_ids))

        try:
            question_sequences = list(_iter_sequences(data["question_id"]))
        except KeyError as exc:
            raise DatasetError(f"{npz_path} is missing the 'question_id' array") from exc

        exercise_to_concepts: Dict[int, set[int]] = {}
        concept_to_exercises: Dict[int, set[int]] = {}
        for question, concept in _iter_pairs(question_sequences, skill_sequences):
            if question < 0 or concept < 0:
                continue
            exercise_to_concepts.setdefault(question, set()).add(concept)
            concept_to_exercises.setdefault(concept, set()).add(question)

        exercises = len(exercise_to_concepts)
        if exercises == 0:
            raise DatasetError(
                f"Dataset '{name}' has no valid exercise identifiers. Ensure question_ids are present."
            )

        avg_concepts_per_exercise = float(
            np.mean([len(concepts) for concepts in exercise_to_concepts.values()])
        )
        avg_exercises_per_concept = float(
            np.mean([len(exercises) for exercises in concept_to_exercises.values()])
        ) if concept_to_exercises else 0.0

    return DatasetStats(
        name=name,
        learners=learners,
        exercises=exercises,
        concepts=concepts,
        interactions=interactions,
        avg_concepts_per_exercise=avg_concepts_per_exercise,
        avg_exercises_per_concept=avg_exercises_per_concept,
        avg_interaction_length=avg_interaction_length,
    )


def _format_value(value: float | int) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:.2f}"


def _render_markdown(datasets: Sequence[DatasetStats]) -> str:
    headers = ["Statistics"] + [stats.name for stats in datasets]
    rows = [
        ("#Learners", [stats.learners for stats in datasets]),
        ("#Exercises", [stats.exercises for stats in datasets]),
        ("#Knowledge concepts", [stats.concepts for stats in datasets]),
        ("#Interaction records", [stats.interactions for stats in datasets]),
        ("Avg. concepts per exercise", [stats.avg_concepts_per_exercise for stats in datasets]),
        ("Avg. exercises per concept", [stats.avg_exercises_per_concept for stats in datasets]),
        ("Avg. interaction length", [stats.avg_interaction_length for stats in datasets]),
    ]

    header_line = " | ".join(headers)
    separator_line = " | ".join(["---"] * len(headers))
    lines = [header_line, separator_line]
    for label, values in rows:
        formatted = [_format_value(v) for v in values]
        lines.append(" | ".join([label] + formatted))
    return "\n".join(lines)


def _render_csv(datasets: Sequence[DatasetStats], delimiter: str = ",") -> str:
    buffer = io.StringIO()
    headers = ["Statistics"] + [stats.name for stats in datasets]
    rows = [
        ("#Learners", [stats.learners for stats in datasets]),
        ("#Exercises", [stats.exercises for stats in datasets]),
        ("#Knowledge concepts", [stats.concepts for stats in datasets]),
        ("#Interaction records", [stats.interactions for stats in datasets]),
        ("Avg. concepts per exercise", [stats.avg_concepts_per_exercise for stats in datasets]),
        ("Avg. exercises per concept", [stats.avg_exercises_per_concept for stats in datasets]),
        ("Avg. interaction length", [stats.avg_interaction_length for stats in datasets]),
    ]
    writer = csv.writer(buffer, delimiter=delimiter)
    writer.writerow(headers)
    for label, values in rows:
        formatted = [
            f"{float(v):.6f}" if not float(v).is_integer() else str(int(v))
            for v in values
        ]
        writer.writerow([label] + formatted)
    return buffer.getvalue().strip()


def _render_table(datasets: Sequence[DatasetStats], fmt: str) -> str:
    if fmt == "markdown":
        return _render_markdown(datasets)
    if fmt == "csv":
        return _render_csv(datasets, delimiter=",")
    if fmt == "tsv":
        return _render_csv(datasets, delimiter="\t")
    raise ValueError(f"Unsupported format: {fmt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute headline statistics for SRC-compatible NPZ datasets and render them "
            "as a markdown table (default) or delimited text."
        )
    )
    parser.add_argument(
        "--dataset",
        metavar="NAME=PATH",
        action="append",
        required=True,
        help=(
            "Dataset specification mapping a display name to either an NPZ file or a directory. "
            "When a directory is supplied the script looks for NAME.npz inside it."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv", "tsv"),
        default="markdown",
        help="Output format (markdown table, CSV, or TSV)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats: List[DatasetStats] = []
    for dataset_spec in args.dataset:
        name, raw_path = _parse_dataset_argument(dataset_spec)
        npz_path = _resolve_npz_path(name, raw_path)
        dataset_stats = _compute_dataset_statistics(name, npz_path)
        stats.append(dataset_stats)
    print(_render_table(stats, args.format))


if __name__ == "__main__":
    main()
