# Repeated cut-in robustness evaluation

Complete all controlled hard-constraint cases before any soft rerun. Only independently confirmed infeasible cases are rerun from the initial state with common slack penalties. Select each successful hard trajectory or its complete soft rerun. Simulate all-HDVs baselines afresh.

15 random IDM profiles x four braking magnitudes x six timings x three methods = 1080 final cases. Evaluate 0--39.96 s after the 50-step initialization. Each final case uses its successful hard trajectory or its complete soft rerun. All-HDVs results are generated in this batch.

case_sources.csv records the selected stage; case_metrics.csv contains the per-case metrics; reviewer_summary.csv contains the 90-run comparisons and 95% profile-cluster bootstrap intervals. Spacing violation duration is averaged only over runs with spacing violations. Slack activation counts successful control solves with slack above 0.001.
