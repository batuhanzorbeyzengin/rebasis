`migrate` has been run against a million-row table, and half of the ROADMAP's largest admission moves with it.

That admission was one sentence: *everything is tested on hundreds of records, not millions, and nobody has yet pointed `migrate` at an index they could not rebuild.* The first half is now measured, because pgvector is the first backend where a table of that size can be stood up in minutes.

1,000,000 rows at 384 dimensions, `spikes/pgvector_scale.py`:

| table | migrated | seconds | records/s | peak traced memory |
|---|---|---|---|---|
| 20,000 | 5,000 | 10.5 | 478 | **7.3 MB** |
| 1,000,000 | 5,000 | 14.1 | 354 | **7.3 MB** |
| 1,000,000 | 100,000 | 342.6 | 292 | 17.4 MB |

**The middle row is the finding.** The same 5,000 records against a table fifty times larger peak at exactly the same 7.3 MB. Peak memory is a function of how many records are *enqueued*, not of how many are in the table — which is the `O(batch × d)` streaming contract the store protocol has always claimed, measured rather than argued from a docstring. The third row's extra 10 MB is 95,000 more queued ids at about 105 bytes each, which is a Python string in a list.

The shadow scaled with it and stayed correct: 100,000 records, 153.6 MB on disk, every migrated record covered, a sample read back cleanly.

Two limits stated with it. The table carried **no index on the vector column**, so these are the write path alone — index maintenance on `UPDATE` is real and measured separately. And it is still not somebody's index: a synthetic million rows on a shared machine is not a production database, and the half of the sentence that needs their data and their backup is unchanged.

One practical finding from standing the table up, which rebasis does not do and a user does: building an IVFFlat index over a million 384-dimensional vectors asks for 76 MB and PostgreSQL's default `maintenance_work_mem` is 64, so the build fails after the whole table has loaded. `SET maintenance_work_mem = '1GB'` for that session.
