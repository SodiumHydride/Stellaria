"""Background jobs: queue, runner threads, GPU scheduling, cancellation, logs."""

from .manager import Job, JobContext, JobManager

__all__ = ["Job", "JobContext", "JobManager"]
