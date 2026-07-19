# 1N Non-deterministic A/B Divergence Points

Last updated: 2026-07-04

Comparison rule:
- Align by `completed step` index (`step 0..N`).
- Divergence point = first step where logged `loss` differs between A and B.

## Recorded results

| Setup | A job | B job | Common steps compared | First divergence step | A loss @ divergence | B loss @ divergence | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| MoE dense_matmul (`ck1k-md-nd`) | `19747` | `19748` | 1000 | **41** | 11.225 | 11.226 | Complete |
| MoE sparse_matmul (`ck1k-ms-nd`, 200-step) | `19753` | `19754` | 200 | **34** | 9.929 | 9.931 | Complete |
| Dense (`ck1k-nd-a2` vs `ck1k-nd-b200`) | `19749` | `19756` | 3 | **2** | 10.831 | 10.660 | Complete (auto-stopped on divergence) |

## Notes

- Earlier dense pair `19743/19744` did not produce train steps (Docker pull failure), so no divergence step can be defined for that pair.
- `19756` live monitor detected divergence and auto-stopped the job immediately after detection.
