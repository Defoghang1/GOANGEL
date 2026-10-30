import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from registry import STAGES_ORDER

HEADER = """name: GOANGEL Sequential

on:
  workflow_dispatch:
    inputs:
      seasons_lookback:
        default: "3"
        type: string
      enrich_xg:
        default: "true"
        type: string
      workers_per_key:
        default: "4"
        type: string
      max_workers:
        default: "256"
        type: string
      rate_limit_per_minute:
        default: "20"
        type: string
      max_attempts_per_model:
        default: "15"
        type: string
      force_stage:
        default: ""
        type: string
      is_retry:
        default: "false"
        type: string

permissions:
  contents: write
  actions: write

concurrency:
  group: goangel-sequential
  cancel-in-progress: true

jobs:
"""

WITH = """    with:
      stage: {stage}
      seasons_lookback: ${{{{ inputs.seasons_lookback }}}}
      enrich_xg: ${{{{ inputs.enrich_xg }}}}
      workers_per_key: ${{{{ inputs.workers_per_key }}}}
      max_workers: ${{{{ inputs.max_workers }}}}
      rate_limit_per_minute: ${{{{ inputs.rate_limit_per_minute }}}}
      max_attempts_per_model: ${{{{ inputs.max_attempts_per_model }}}}
      is_retry: ${{{{ inputs.is_retry }}}}
    secrets: inherit
"""


def job(stage, previous):
    lines = [f"  {stage}:", f'    name: "{stage}"']
    if previous is None:
        lines.append("    if: inputs.force_stage == '' || inputs.force_stage == 'prepare'")
    else:
        lines += [
            f"    needs: {previous}",
            "    if: |",
            "      always() && !cancelled() &&",
            "      (",
            f"        needs.{previous}.result == 'success' ||",
            f"        (inputs.force_stage == '{stage}' && needs.{previous}.result != 'failure')",
            "      )",
        ]
    lines.append("    uses: ./.github/workflows/_stage_runner.yml")
    return "\n".join(lines) + "\n" + WITH.format(stage=stage)


def main():
    parts = [HEADER]
    previous = None
    for stage in STAGES_ORDER:
        parts.append(job(stage, previous) + "\n")
        previous = stage
    parts.append(
        "  release:\n"
        '    name: "release"\n'
        f"    needs: {previous}\n"
        "    if: |\n"
        "      always() && !cancelled() &&\n"
        f"      (needs.{previous}.result == 'success' || inputs.force_stage == 'release')\n"
        "    uses: ./.github/workflows/release.yml\n"
        "    with:\n"
        '      version: ""\n'
        "    secrets: inherit\n"
    )
    out = Path(__file__).resolve().parents[1] / "workflows" / "train.yml"
    out.write_text("".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
