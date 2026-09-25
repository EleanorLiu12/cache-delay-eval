# Cache Hit and TTFT Trace Patterns

**Objective: identify workload patterns in which higher actual cache-hit fractions coincide with higher time to first token (TTFT), and explain the conditions under which this association appears.**

The study uses request sequences on a single LLM server. We generate different session structures, replay them at fixed arrival times, and measure each request's `cached_tokens / prompt_tokens` and TTFT.

| Question | Entry point |
| --- | --- |
| What are the research question, metrics, and next steps? | [Research scope](docs/research.md) |
| What did the first GPU experiment measure and find? | [GPU experiment report](results/cloudlab-pattern-03/report.md) |
| How do I install the tools, generate traces, and reproduce the experiment? | [Run guide](docs/pattern-pilot.md) |
| How were candidate traces selected? | [Candidate screening report](results/pattern-screen/report.md) |
| How are requests represented? | [Trace format](schemas/README.md) |

Source code is in `src/cache_delay_eval/`, experiment settings in `config/pattern-pilot.json`, raw measurements and analysis in `results/`, and plotting scripts in `scripts/`. Findings are maintained in the experiment reports.
