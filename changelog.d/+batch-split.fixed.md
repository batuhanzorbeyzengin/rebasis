A rejected batch is retried and then split, instead of failing whole.

`StoreWriteFailed` declares itself transient — a store that refused a write because a node was rebalancing usually takes it a moment later — and `retry_transient` was written for exactly that and called from nowhere. It is now wired onto the migration's write: three attempts, exponential backoff with jitter, every attempt after the first logged.

When retrying does not help, the batch is halved and each half written separately, recursively. Before this a rejected batch was marked `FAILED` whole, so one oversized payload or one id the store would not take cost its two hundred and fifty-five neighbours a place in the failed list and a second pass on the next `resume`. Nothing was lost — the queue is the checkpoint — but the operator had 256 records to look at instead of one.

Two bounds keep the cost honest, and both were set by measurement rather than taste. **Splitting stops after four levels**, because a store that is simply unreachable fails every half and splitting all the way down costs 511 writes to learn what the first one already said. **The halves are not retried**: with the retry on every node, isolating one bad record from a batch of sixteen took 23 seconds, almost all of it backing off from a refusal the batch's own three attempts had already settled. Removing it took the same test file from 216 seconds to 22.

`docs/guides/migration.md` said the job stops when a batch fails. It does not, and did not — the loop continues to the next batch and a run finishes with a count of failures rather than at the first one. That section now describes the retry, the split, and both bounds.

A run's `processed` count is now what landed rather than what was offered. Those were the same number while a batch was all-or-nothing; they are not any more, and reporting the batch size would have had a run claim it processed records the queue holds as `FAILED`.
