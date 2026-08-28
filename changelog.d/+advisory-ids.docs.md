`SECURITY.md` names each open advisory by its GHSA identifier, which is the one a Dependabot alert shows.

The table cited two of the five as `PYSEC-…` and three as `GHSA-…`, so matching an alert on the repository's security tab to the assessment written for it meant looking up an alias first. Every row now leads with the GHSA identifier and carries the PYSEC and CVE aliases beside it. The assessments themselves are unchanged, and so is the conclusion: five advisories, no fixed version upstream for any of them, none reachable through the way rebasis uses either package.
