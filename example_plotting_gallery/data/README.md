# Synthetic response-panel fixtures

Every value, identifier, group assignment, feature name, and effect estimate in
this directory is synthetic. These files do not reproduce or summarize a
clinical-study cohort.

- `synthetic_expression.csv` contains one row per synthetic sample and feature.
- `synthetic_effects.csv` contains one supplied synthetic effect and confidence
  interval per feature.

Response group, subtype, and cohort are sample-level annotations and remain
constant for each `sample_id` across all feature rows.

The confidence intervals are plotting inputs. They were not estimated from the
expression table.
