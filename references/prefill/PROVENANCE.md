# Provenance and preservation

The measured baseline is `8f09ee41b461f5cbdbac01594b24f1c7578b040e`. The experimental source-only digest is
`b710c5b3774bb234c0d3f7cf037871403dd92936a05f90b6069c4a2eac9f48fa`. Each final run's manifest records exact
source-file, harness, workload and isolated-cache hashes. Copied-source runtime revision metadata is null;
no current commit is substituted for it. Both arms used the preserved cache seed; its old tuning key did
not select entries for the current kernel version.

The candidate remains in an ignored recovery archive. Its [archive manifest](data/archive-manifest.json.gz)
hashes 618 checked files, including sources, tests, harness, cache seed, correctness and all six final results.
The full contaminated raw reports and private activity logs also remain in the local experiment directories;
[attempts.json](attempts.json) identifies them by relative source ID and SHA256 without publishing process
names. The public datasets preserve accepted raw values and all original statistical verdicts.

| Recovery artifact | SHA256 |
|---|---|
| archive manifest | `942225981a57812e739f26012575f4d7204a8ae1b92329c3aa40369b72b2689f` |
| experimental candidate patch | `49b4918f551f7bf9c9767f8dbfe50dc1af2d4e9859c7e98d7a44630c7d16af81` |
| exact measured-source recovery patch | `b3f6e691e297ff31c04c649567da2ca418160694b7da8658a5694136e8476941` |

The first patch contains the 11 experimental source/test changes and retains an unrelated precision cleanup
in current main. The second additionally restores the measured precision module for exact reconstruction.
Neither patch was applied to the shipped library. Source hashes in the run manifests are authoritative for
what was measured.

## Measurement scope

The final gate has 36 fresh cells and 24 growing views. Each fresh cell has 32 timed requests per arm in total,
with four paired rounds. Growing includes the first request, steady requests and whole-session wall time;
these overlap two session configurations, each repeated in four paired rounds, rather than independent
workloads. All six jobs completed their declared
coverage. All fresh jobs rejected; all growing jobs promoted. The overall decision is reject and archive.

A paired unit with recorded external CPU use above 50% was excluded and retained, then repeated with the
same requests. Monitoring failures also invalidated a pair. No clean slow observation was removed. One-second
sampling can miss very short-lived processes and the final CPU delta of an exited process; the recorded
monitoring limitations and numerical activity data remain in the complete result datasets.

## Reproducible publication audit

`publish.py --source DIRECTORY` deterministically recreates the gzip datasets and generated tables from the
retained experiment; it never loads models or runs timing. Gzip timestamps are fixed. `audit.py` checks file
hashes, every recorded numerical/boolean leaf, complete final-cell counts and gate statuses. With `--source`,
it also checks source hashes, direct value equality and every retained attempt's raw-file/activity-log hashes.
Normalization derives original path prefixes from immutable source metadata, so a later folder move leaves
raw files untouched and remains auditable.

Only nonmeasurement environment metadata is omitted; those omitted leaves are expressly outside the
numerical-equivalence claim. All measurement numbers, probabilities, token counts, raw latencies, activity
shares, flags and original verdicts remain unchanged. String normalization makes paths portable and busy
labels anonymous. Displayed time-change percentages are derived from preserved raw samples; original ratios
and confidence intervals remain in the datasets.
