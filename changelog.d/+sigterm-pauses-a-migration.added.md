`rebasis migrate` catches SIGTERM and stops at the next batch boundary instead of being killed where it stands.

A long migration does not run on a laptop with somebody watching it. It runs as a Kubernetes `Job`, an Airflow task or an Argo step, and all three end a process the same way — SIGTERM, a grace period, then SIGKILL. There was no handler for either signal, so the default applied: immediate termination, typically mid-batch, leaving the store holding records that were never read back and compared. The checkpoint made that survivable; it did not make it clean.

The signal is now the same request `rebasis pause` makes, arriving from a different direction. The run stops at a boundary, records the job as paused, names the signal that asked, and `rebasis resume <job-id>` picks it up. SIGINT is caught too, so Ctrl-C asks rather than aborts.

**The second signal is not caught.** The handler restores the default before it returns, so a supervisor escalating still stops the process at once — a graceful stop that cannot be interrupted is a hang with better manners.

**The grace period has to outlast a batch.** A batch that takes longer than `terminationGracePeriodSeconds` is still killed part-way and no handler changes that; raise the period or lower `--batch`. The new [production guide](https://batuhanzorbeyzengin.github.io/rebasis/guides/operations/) has the numbers, along with exit codes, secrets, offline installation and what the state lock does and does not coordinate.

The CLI installs the handler, not `MigrationEngine`. A library caller keeps their own signal handling: taking SIGTERM away from the application embedding you is not a library's to take.
