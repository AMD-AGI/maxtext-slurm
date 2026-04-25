# rules.md

## Hard rules

### Research
- Do not treat external claims as verified facts for AMD unless validated or clearly marked as external reference
- Every important claim must include a source or code evidence

### Code investigation
- A config flag is not considered effective unless its usage path is traced to actual execution
- Always distinguish between exposed setting and effective behavior

### Experiments
- No determinism claim without explicit reproducibility criteria
- Every experiment must record:
  - git commit
  - docker image
  - node count
  - config file
  - environment variables
  - hardware
  - seed
  - relevant flags
- Only one major factor should change per experiment unless explicitly marked as combined experiment
- Performance comparisons must use the same workload and same scale

### Writing
- Do not claim full determinism unless the exact scope is stated
- Every claim in the blog must be labeled as one of:
  - verified
  - inferred
  - external reference
  - open question