"""One pipeline, raw data -> submission (plans/PIPELINE_DESIGN_2026_09_10.md).

    config -> ingest -> lanes (route, model plug-ins) -> assemble -> validate -> write

P2a: a skeleton around the shipped, tested functions with no behaviour change. Entry point:
`python -m prc.pipeline run --config configs/pipeline.yaml --out <dir> [--dry-run]`. Importing this package
imports nothing heavy; each module loads the scripts it wraps lazily (prc.pipeline.legacy).
"""
