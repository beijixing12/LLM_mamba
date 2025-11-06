from __future__ import annotations

"""Prepare ASSISTments problem log datasets with ``problem_log_id`` ordering.

The logic mirrors :mod:`prepare_assist2009` but tweaks column defaults so that
newer ASSISTments exports (which label concepts using the ``skill`` column and
order interactions via ``problem_log_id``) can be converted without editing the
original script.  The resulting ``.npz`` and concept-mapping ``.csv`` files are
identical in structure to those produced by the original ASSISTments helper.
"""

import argparse
import json
from pathlib import Path

from prepare_assist2009 import (
    load_raw_dataset,
    handle_missing_values,
    drop_missing_skill_names,
    build_examples,
    reindex_concepts,
    write_npz,
    write_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ASSISTments problem logs into the SRC NPZ format",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw ASSISTments problem log CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the processed files (defaults to the CSV directory)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="assist_problem_logs",
        help="Base name used for the generated .npz file",
    )
    parser.add_argument(
        "--order-column",
        type=str,
        default="problem_log_id",
        help="Column used to sort a learner's history",
    )
    parser.add_argument("--question-column", type=str, default="assistment_id")
    parser.add_argument("--attempt-column", type=str, default="attempt_count")
    parser.add_argument("--correctness-column", type=str, default="correct")
    parser.add_argument("--first-time-column", type=str, default="ms_first_response")
    parser.add_argument("--response-time-column", type=str, default="ms_first_response")
    parser.add_argument("--hint-column", type=str, default="hint_count")
    parser.add_argument("--skill-id-column", type=str, default="skill_id")
    parser.add_argument("--skill-name-column", type=str, default="skill")
    parser.add_argument(
        "--missing-strategy",
        choices=("impute", "drop"),
        default="impute",
        help=(
            "How to handle missing values in interaction columns. 'impute' replaces them with safe defaults "
            "while 'drop' removes affected interactions before sequence construction."
        ),
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        help="Encoding of the raw CSV file. When omitted, the script will try several common options",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=None,
        help="Optional custom path for the concept-id mapping CSV",
    )
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help="If set, create a JSON file with dataset statistics for quick inspection",
    )
    return parser.parse_args()


def process_problem_logs(
    *,
    input_path: Path,
    output_dir: Path | None,
    dataset_name: str,
    order_column: str,
    question_column: str,
    attempt_column: str,
    correctness_column: str,
    first_time_column: str,
    response_time_column: str,
    hint_column: str,
    skill_id_column: str,
    skill_name_column: str,
    missing_strategy: str,
    encoding: str | None,
    mapping_file: Path | None,
    write_summary: bool,
) -> None:
    input_path = input_path.expanduser().resolve()
    if output_dir is None:
        output_path = input_path.parent
    else:
        output_path = output_dir.expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_dataset(
        input_path,
        order_column=order_column,
        encoding=encoding,
    )

    cleaned_df, missing_report = handle_missing_values(
        raw_df,
        strategy=missing_strategy,
        drop_columns=("user_id", order_column),
        fill_defaults={
            question_column: -1,
            attempt_column: 0,
            correctness_column: 0,
            first_time_column: 0,
            response_time_column: 0,
            hint_column: 0,
        },
    )

    cleaned_df, dropped_missing_skill_names = drop_missing_skill_names(
        cleaned_df,
        skill_name_column=skill_name_column,
    )
    examples, mapping = build_examples(
        cleaned_df,
        question_column=question_column,
        attempt_column=attempt_column,
        correctness_column=correctness_column,
        first_time_column=first_time_column,
        response_time_column=response_time_column,
        hint_column=hint_column,
        skill_id_column=skill_id_column,
        skill_name_column=skill_name_column,
    )

    examples, reindexed_mapping, missing_name_ids = reindex_concepts(
        examples,
        mapping,
    )

    dataset_path = output_path / f"{dataset_name}.npz"
    if mapping_file is None:
        mapping_path = output_path / "knowledge_concept_mapping.csv"
    else:
        mapping_path = mapping_file
        if not mapping_path.is_absolute():
            mapping_path = output_path / mapping_path

    write_npz(dataset_path, examples, reindexed_mapping)
    write_mapping(mapping_path, reindexed_mapping)

    print(f"Saved processed dataset to {dataset_path}")
    print(f"Saved concept mapping to {mapping_path}")

    if missing_report["dropped_rows"]:
        drop_details = [
            f"{col}: {count}"
            for col, count in sorted(missing_report["dropped_by_column"].items())
            if count
        ]
        if drop_details:
            print(
                "Dropped %d rows due to missing values (%s)"
                % (missing_report["dropped_rows"], ", ".join(drop_details))
            )
        else:
            print(
                "Dropped %d rows due to missing values"
                % missing_report["dropped_rows"]
            )
    if missing_report["imputed_by_column"]:
        print(
            "Imputed missing values in: %s"
            % ", ".join(
                f"{col} ({count})"
                for col, count in sorted(missing_report["imputed_by_column"].items())
            )
        )

    if dropped_missing_skill_names:
        print(
            "Dropped %d rows due to missing skill names"
            % dropped_missing_skill_names
        )

    if missing_name_ids:
        formatted = ", ".join(str(new_id) for new_id in missing_name_ids)
        print(
            "Warning: %d concepts lack names in the source data: %s"
            % (len(missing_name_ids), formatted)
        )

    if write_summary:
        total_interactions = sum(example.seq_len for example in examples)
        summary = {
            "num_users": len(examples),
            "concepts": len(reindexed_mapping),
            "total_interactions": int(total_interactions),
            "max_seq_len": max((example.seq_len for example in examples), default=0),
            "mean_seq_len": (
                float(total_interactions) / len(examples)
                if examples
                else 0.0
            ),
            "skipped_interactions": int(len(raw_df) - total_interactions),
            "missing_value_report": missing_report,
            "output_file": str(dataset_path),
            "dropped_missing_skill_names": dropped_missing_skill_names,
            "concepts_missing_names": [
                {"id": new_id}
                for new_id in missing_name_ids
            ],
        }
        summary_path = output_path / "dataset_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf8")
        print(f"Saved dataset summary to {summary_path}")


def main() -> None:
    args = parse_args()
    process_problem_logs(
        input_path=args.input,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        order_column=args.order_column,
        question_column=args.question_column,
        attempt_column=args.attempt_column,
        correctness_column=args.correctness_column,
        first_time_column=args.first_time_column,
        response_time_column=args.response_time_column,
        hint_column=args.hint_column,
        skill_id_column=args.skill_id_column,
        skill_name_column=args.skill_name_column,
        missing_strategy=args.missing_strategy,
        encoding=args.encoding,
        mapping_file=args.mapping_file,
        write_summary=args.write_summary,
    )


if __name__ == "__main__":
    main()

