"""Thin wandb wrapper so train/eval/sweep share one logging convention.

Everything here is a no-op when `--wandb` is off, so the scripts stay runnable
without a network connection or a wandb account.
"""

from typing import Optional

DEFAULT_ENTITY = "cs23b035-iit-madras"
DEFAULT_PROJECT = "cs6886-a2-mobilenetv2"


def init_run(
    enabled: bool,
    project: str = DEFAULT_PROJECT,
    entity: Optional[str] = DEFAULT_ENTITY,
    name: Optional[str] = None,
    config: Optional[dict] = None,
    job_type: Optional[str] = None,
    group: Optional[str] = None,
):
    """Start a wandb run, or return None when logging is disabled."""
    if not enabled:
        return None

    import wandb

    run = wandb.init(
        project=project,
        entity=entity or None,
        name=name,
        config=config or {},
        job_type=job_type,
        group=group,
    )
    # use epoch, not wandb's internal step counter, as the x-axis
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("test/*", step_metric="epoch")
    run.define_metric("lr", step_metric="epoch")
    run.define_metric("best/*", step_metric="epoch")
    return run


def log_curves(run, history: dict, key: str = "curves/loss_accuracy") -> None:
    """Attach the rendered loss/accuracy figure to the run as an image panel."""
    if run is None:
        return
    import wandb

    from .plotting import curves_figure

    fig = curves_figure(history)
    run.log({key: wandb.Image(fig)})
    import matplotlib.pyplot as plt

    plt.close(fig)


def log_sweep_table(run, rows: list, columns: list, key: str = "compression/sweep") -> None:
    """Log the sweep as one table -- the Parallel Coordinates chart itself is
    built from the individual per-config runs, this is just a copy for reference."""
    if run is None:
        return
    import wandb

    run.log({key: wandb.Table(columns=columns, data=rows)})
