# read Codex Instructions @ /home/ubuntu/codex.md
Codex Instructions

no low-value helper functions
no low-value guards

PURPOSE
- This repository contains bioinformatics pipelines.

SESSION WORKFLOW (DEFAULT)
- Default for each new session: do not modify any files immediately.
- First, propose a plan and a short diff summary (what files, what will change, why).
- Only proceed to edit files after explicit approval for this session.
- Approval applies only to the current session; next session reverts to default.

SCIENTIFIC INTEGRITY (HARD RULES)
- Do not change statistical models, thresholds, or biological assumptions unless explicitly requested.
- Preserve scientific intent and numerical behavior by default (including edge cases).

ENGINEERING CONSTRAINTS
- Preserve public APIs and function signatures.
- Prefer minimal, reviewable diffs; modify only what is required to meet the objective.
- Prefer vectorized numpy/pandas over Python loops when it improves performance and memory behavior.
- Avoid unnecessary abstractions and avoid refactoring unrelated code.
- Assume large datasets and memory pressure.
- Python >= 3.10.
- No new external dependencies unless explicitly approved (per session).

QUALITY BAR
- Output must be deterministic.
- Add docstrings and type hints when helpful, but do not churn formatting.
- Explain non-obvious changes inline as comments.
- Call out assumptions and risks explicitly (in comments and/or the plan).
- Add or update tests only if behavior changes or if new functionality is introduced.

COMMUNICATION & REVIEW (IMPORTANT)
- When presenting findings, recommendations, risks, options, possible actions, plans, diff summaries, or questions, always use ONE numbered list.
- Numbering must be globally unique and sequential within the message (1, 2, 3, ...), not restarted across sections.
- Do not use unnumbered bullets for actionable content.
- Each numbered item must be independently actionable or answerable.
- If you want sections, prefix inside the item, e.g.:
  1) PLAN: ...
  2) DIFF: ...
  3) QUESTION: ...
- Expect responses of the form: "yes to #1, no to #2, explain #3 more" and adjust accordingly.
- When requesting approval, ask for approval per item (or per group of item numbers), not a generic "yes".
- When requesting approval when possible use the green aproval bottom used by codex

TASK PROMPT TEMPLATE (FILL IN PER REQUEST)

ROLE
You are a senior software engineer with deep expertise in [bioinformatics | data engineering | ML | backend].

CONTEXT
This code is part of [pipeline | library | analysis] used for [purpose].
Inputs are [size, format, constraints].
Downstream consumers are [humans | APIs | models].

OBJECTIVE
[Primary goal: refactor | optimize | debug | extend | productionize].

CONSTRAINTS
- Maintain public API and function signatures
- Do not change statistical or biological assumptions
- Optimize for [runtime | memory | readability | numerical stability]
- No new external dependencies (unless explicitly approved)

NON-GOALS
- Do not refactor unrelated files
- Do not introduce abstractions unless necessary
- Do not change behavior unless explicitly stated

QUALITY BAR
- Deterministic output
- Clear docstrings and type hints where valuable
- Edge cases handled

DELIVERABLE
- Provide a plan + diff summary before editing files (unless approved for this session)
- Provide a minimal diff
- Explain non-obvious changes inline as comments
- Call out assumptions or risks explicitly
