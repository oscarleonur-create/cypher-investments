"""The always-on daemon: scheduled jobs, an event stream, and ingest watermarks.

Phase 1 is the spine only — a supervised asyncio loop that knows when the US
equity market is open, runs jobs on schedule, survives its own crashes, and
leaves a durable trace. The jobs it runs are deliberately empty; ingest and
reasoning land in later phases.
"""
