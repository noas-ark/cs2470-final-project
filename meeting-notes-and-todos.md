# Meeting Notes & TODOs

## 10/21 — Shiham Arch Project Discussion

### Experiment Design: Independent and Dependent Variables
- Primary independent variable: model family/configuration
  - Start with ~5 representative open-source decision models, not all 600
  - Group by flavor: fine-tuned decision models, regular-LLM-like, diffusion-based
  - Pick one representative per group to avoid experiment bloat
- Key dependent variable to start: GPU utilization (via nvidia-smi, no Nsight needed)
  - Expand to other DVs (benchmark accuracy, clock time) iteratively
  - Keep DVs consistent across runs for reproducibility
- Fixed controls for first pass: temperature, context length, environment
  - These can become levels of the IV later, but not yet
- Avoid experiment slop: only add new IVs when there's a clear hypothesis

### Experiment Tracking and Data Schema
- Each run stored as a JSON with two top-level keys: params and results
  - params: model, GPU type, environment, context length, temperature, etc.
  - results: benchmark scores, GPU utilization, etc.
- Schema is intentionally flexible: newer runs can have more result types than older ones
- Every experiment committed to Git, config + output in the same commit
  - Git commit = source of truth for reproducibility
  - CI check on PRs to enforce required fields are present (no missing IVs or DVs)
- Avoid separate IV/DV lists that can drift out of sync; nest results under model key instead

### CLI and Repo Structure
- Shared GitHub repo as source of truth; clone to login node
- CLI wraps common commands so neither person is juggling raw shell commands
- tmux used by default to persist sessions across GPU node disconnects
- Claude can be installed directly on the login node (use school accounts)
  - Preferred workflow: Claude Code on login node runs experiments end-to-end via terminal
  - Local machine used for iteration with Claude when GPU not needed

### Compute Environment: Slurm vs. Modal
- Slurm: srun provisions GPU from AWS/Kubernetes; takes time to allocate
  - Login node has minimal memory but persists files; GPU node is ephemeral
  - Unlimited screwing around; good for early exploration
- Modal: finite credits but developer-friendly; closer to industry workflow
  - Worth emailing Modal contacts for more credits (possible joint blog post angle)
- Goal: single env-var switch (environment=slurm or environment=modal) routes identically
  - Claude Code can build the mock API routing layer quickly

### Model Download and Test Sequencing
- Bottleneck is network: downloading weights to GPU HBM, not compute
  - Each model swap: ~15 min download, so 5 models ≈ ~1 hour total
  - Load one model, run all tests, then swap to next (not the reverse)
- Parallelization possible across multiple GPUs but adds complexity
- Prioritize selecting models with meaningfully different hardware profiles, not just accuracy

### Benchmarking Approach and Roofline Insights
- First milestone: consistent, reproducible results across both team members on Slurm
- Use one fixed benchmark (e.g. "JevBench" equivalent) for the initial sweep
- GPU utilization is the primary signal: tensor cores vs. FP32 cores vs. memory-bound vs. network-bound
- Roofline model context: roofline position shifts with context length; weight-dominated vs. activation-dominated regimes behave differently
- Decision models vs. regular LLMs: similar accuracy possible but very different GPU usage profiles
  - Same benchmark perf + different GPU profile = interesting analysis target
- Pareto frontier (cost vs. accuracy): decision model combos can cross GPT-5 accuracy at lower cost and higher speed

### Decision Model Findings (Prior Work)
- Decision models notably weak at counterfactual reasoning, especially physical before/after scenarios
- OpenJev variants have vastly different characteristics; not interchangeable
- Cascade/combo approach (decision model + VLM) traces a Pareto frontier vs. GPT-5 baseline
  - Crossed GPT-5 accuracy threshold at lower cost and significantly better speed

## TODOs

- [ ] **TODO 1: Set up repo, README, and basic experiment flow.** Pick one leading open-source model, wire NVIDIA SMI utilization tracking, and get a first run committed with config + output in the same Git commit.
- [ ] **TODO 2: Narrow down to ~5 representative models to test first.** Select based on architectural diversity and expected hardware profile differences, not random sampling from the full 600.

(Stretch, not yet committed to: add CI enforcement for required experiment fields on PRs, to prevent IV/DV/results schema drift.)
