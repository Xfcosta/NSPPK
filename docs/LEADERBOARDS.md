# Leaderboard comparisons

`nsppk_leaderboard.check_leaderboard(...)` fetches an HTML leaderboard, returns its
rows, computes the mean and standard deviation of the listed scores, and plots
the supplied result beside the leaderboard entries. It recognizes scores
written as `mean ± standard deviation` or `mean +/- standard deviation`.

```python
from nsppk_leaderboard import check_leaderboard

comparison = check_leaderboard(
    "https://ogb.stanford.edu/docs/leader_graphprop/#ogbg-molhiv",
    score=0.86,
    section="ogbg-molhiv",
    metric_column="Test ROC-AUC",
    result_name="NSPPK + random forest",
    higher_is_better=True,
    comparison_note="Descriptive only: evaluation split differs from official OGB.",
)

rows = comparison["results"]
summary = comparison["summary"]
comparison["figure"].show()
```

`results` contains every parsed table row with its original cell text and
normalized `_score` and `_reported_std` fields. `summary` contains the
leaderboard mean and standard deviation, the supplied score's z-score,
percentile, and a rank based on the listed mean scores. That rank is
descriptive; compare evaluation protocols and dataset splits before treating it
as a valid leaderboard placement. A local random split, for example, is not
directly comparable to an official scaffold split.

This utility uses Python's standard library for fetching and parsing HTML and
the package's existing Matplotlib dependency for the plot. It does not submit
or publish results to the leaderboard.
