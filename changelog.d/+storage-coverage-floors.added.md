`storage/shadow.py` and `storage/atomic.py` carry a coverage floor of their own.

These two write and restore the bytes a rollback depends on, and `ROADMAP.md` named 95 and 100 as their targets. `tools/check_coverage_floors.py` enforced neither: the pair was held only by the `src/rebasis/storage/` package floor of 80, which they clear together at 90.6% without either module having to hold anything on its own. A target stated in prose and enforced nowhere is not a target.

Both now have a floor at 90 — a ratchet rather than the goal, set under the measured 93.5% and 93.3% so it gates from the day it lands, and the 95 and the 100 stay where they were as the thing still to reach.
