# Mutation Distribution Analysis — Conceptual Plan

## Goal

Characterize the statistical distribution of SNP mutations introduced by the evolutionary
optimization algorithm, and use that distribution to generate biologically realistic random
baseline mutations for comparison.

---

## Core Idea

Each mutation is an atomic unit described by three properties:

```python
(position, source_base, new_base)
```

We model the joint distribution over these three quantities empirically — by pooling all
observed mutations across all optimized genes — and then sample from that distribution to
generate baselines.

---

## Step 1: Build the Mutation Pool

For each optimized gene, extract all SNPs by comparing the optimized sequence to the
wildtype reference. Each SNP yields one tuple:

- `position`: integer index along the 3000 bp promoter window
- `source_base`: the base in the wildtype sequence at that position (`A`, `C`, `G`, `T`)
- `new_base`: the base introduced by the algorithm

Pool all tuples across all 1000 genes into a single list (~60,000–90,000 entries).

---

## Step 2: Estimate the Empirical Joint Distribution

The joint distribution is estimated directly from the pool.

### **Positional marginal — P(position)**

Count how many mutations occurred at each position across all genes and all sequences.
Normalize by total mutation count. This yields a discrete PMF over positions 0..2999.

Expected density: ~25 observations per position on average — sufficient for direct
empirical estimation without smoothing.

### **Conditional substitution distribution — P(new_base | position, source_base)**

For each (position, source_base) pair, count occurrences of each new_base. Normalize per
group. This captures the position- and context-dependent substitution preferences of the
algorithm.

Note: since the source base at a given position varies across the 1000 genes (different
gene sequences), explicitly conditioning on source_base is necessary to avoid conflating
biochemically distinct events (e.g., A→T and C→T at the same position).

### **Optional diagnostic: transition/transversion ratio**

Compute the fraction of mutations that are transitions (A↔G, C↔T) vs. transversions.
A strong bias here indicates the model has absorbed biological substitution preferences
from the training data.

---

## Step 3: Sample Baseline Mutations for a Given Gene

To generate one random baseline sequence for a specific gene:

1. **Sample mutation count** $k$: draw from the empirical distribution of mutations-per-gene
   (histogram over the 60–90 range observed across the 1000 genes).

2. **Sample positions**: draw $k$ positions without replacement from the positional PMF.
   Without-replacement ensures no two mutations land at the same site.

3. **Sample new bases**: for each sampled position, look up:
   - The wildtype base of the target gene at that position (`source_base`)
   - The conditional distribution `P(new_base | position, source_base)` from the pool
   - If no observations exist for that (position, source_base) combination, fall back to
     the marginal `P(new_base | source_base)` across all positions.

4. Apply the sampled mutations to the wildtype sequence to produce the baseline.

---

## Step 4: Validation

Before using baselines in downstream analysis, verify the sampler is well-calibrated:

- **Count distribution**: sampled $k$ values should match the observed distribution
- **Positional distribution**: histogram of sampled positions should match the empirical
  positional PMF
- **Substitution matrix**: aggregate transition/transversion ratio of sampled mutations
  should match the pool
- **Sanity check**: no two mutations at the same position in any single baseline sequence

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Indels | Excluded | Algorithm only introduces SNPs |
| Pooling | Across all genes | ~75k mutations gives reliable empirical estimates |
| Source base | Included | Genes have different sequences; conflating source bases is incorrect |
| Smoothing | Not required | ~25 obs/position is sufficient for direct empirical sampling |
| Fallback | Marginal P(new_base \| source_base) | Handles unseen (position, source_base) pairs |
| Sampling | Without replacement | No two SNPs at the same position |
