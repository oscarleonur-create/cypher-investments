"""External information ingest: filings, news, and broker facts.

The unit here is not "an article" but a **fact with provenance**. Every item
carries where it came from, when it was published, when we retrieved it, and
how it was tied to a security — because the failure mode this module exists to
prevent is a confident claim with no traceable origin.
"""
