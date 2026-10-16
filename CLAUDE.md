# RegionFlow — working notes for Claude

Current state lives in `STATUS_REVIEW.md` (read first), then `GATE_A_REOPENING_PLAN.md`
(the plan's *Verdict* and *Next: the real work queue* sections) and `GATE_C_REPORT.md`.
`START_HERE.md` is partly stale (it still calls Gate C untouched).

## Model setup (see `.claude/`)
- Main session plans and decides at high effort. Advisor is configured via `advisorModel`
  in `.claude/settings.json`; if that is not honoured, use `/advisor opus` or `--advisor opus`.
- Call the advisor before committing to an approach, when stuck, and before declaring done.
- Delegate at medium effort via the project subagents:
  - `explorer` — read-only code search / tracing. Use it for any sweep across many files.
  - `researcher` — papers, library docs, reference implementations, published numbers.
  - `worker` — a well-specified edit plus its CPU check. Give it files, behaviour, and the check.
- Don't delegate the decision itself or the final review of a diff; do those in the main session.

## Environment
- Cloud containers here have no GPU, no `/data` cache and no `runs/`. Nothing trains or scores
  here. CPU checks: `python transport/pairflow.py --self-test`, `python tests/check_names.py
  scripts/*.py transport/*.py bench/*.py` (needs torch; AST checks do not).
- Never run training/eval commands and report their results; write the command for the user.

## Habits that cost time to learn (from START_HERE.md)
1. Score the ground truth on any new metric before trusting it.
2. Parse, don't grep (`ast`, `torch.load`), when a claim depends on structure.
3. Test the path that actually executes, not a neighbouring function.
4. Read the reference implementation before building from a paper.
5. Make sure the file you ran is the file you edited (print a hash if unsure).
6. Look at the images before recording any number.
7. Pre-register a kill criterion before each experiment; don't answer a negative with a deeper
   diagnostic without deciding what result would end the line.
