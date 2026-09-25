# Incomplete GPU Run

This directory preserves the timeout failure from the first GPU experiment attempt. The heavy trace exceeded the 600-second client timeout, leaving the suite incomplete. It is not used to draw hit–TTFT conclusions.

The subsequent attempt changed only the timeout to 2400 seconds. See the successful run in [cloudlab-pattern-03/report.md](../cloudlab-pattern-03/report.md). Raw requests, server logs, and source snapshots are retained for provenance.
