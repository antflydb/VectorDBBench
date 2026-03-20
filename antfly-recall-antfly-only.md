# Antfly Recall Assessment: Antfly-Only Findings

## Scope

This note is intentionally limited to **Antfly-side** explanations for the low recall seen in `#antfly > vector-db-bench`.

I am **not** treating the original `VectorDBBench` client wiring issues as part of this diagnosis. Those confounders were removed before the rerun summarized below. The question here is narrower: after controlling for the obvious external issues, does the remaining low recall look like an Antfly problem?

## Bottom line

Yes, at this point the low recall looks much more like an **Antfly-side issue** than a benchmark wiring mistake.

After rerunning with the benchmark-side confounders removed, recall stayed essentially unchanged:

- `recall@100 = 0.5008`
- `ndcg = 0.6166`

That means the remaining plausible causes are now mostly inside Antfly itself:

1. a concrete cosine/HBC correctness bug visible in the source-built upstream server
2. hardcoded HBC search defaults that may be too aggressive for this workload
3. dynamic pruning behavior in HBC
4. quantized leaf search
5. possible drift between the running local Antfly binary and the checked-out source/API expectations

## Findings

### 1. The remaining problem is no longer well explained by client wiring

I reran the 50K case after removing the main external confounders and the result remained about the same as before. That materially weakens the hypothesis that recall was low simply because the benchmark client was wired incorrectly.

So the updated diagnosis is:

- the original setup did have avoidable confounders
- but those confounders were **not** the main reason recall ended up near 50%

That pushes the investigation onto Antfly itself.

### 2. The source-built upstream Antfly still fails after receiving unit-normalized cosine vectors

I then reran against a fresh source-built upstream checkout at `~/Documents/antfly/antfly-upstream` after hardening the client further:

- correct cosine `distance_metric`
- explicit cosine normalization on insert/query in the Antfly client
- `need_normalize_cosine()` enabled so `VectorDBBench` normalizes cosine datasets before calling the client

For the first insert batch, the benchmark logged:

- `metric=COSINE`
- `use_cosine=True`
- `first_sqnorm=1.000000`

So at that point the benchmark was definitely sending a unit-normalized cosine vector.

Even with that in place, the upstream source build still panicked inside Antfly during HBC split/reclustering with:

- `vector is not a unit vector`
- stack rooted in `BalancedBinaryKmeans.validateVectors()` via `HBCIndex.splitVectorSet()`

That is the strongest current evidence in this whole investigation. It means there is at least one remaining Antfly-side cosine/HBC correctness problem that is **not** explained by the benchmark simply forgetting to normalize vectors.

### 3. Antfly's dense-vector path is using hardcoded HBC defaults that could plausibly cap recall

The embeddings index builds an HBC index with a fixed configuration in source:

- quantization enabled
- `Episilon2 = 7`
- `BranchingFactor = 168`
- `LeafSize = 168`
- `SearchWidth = 1008`

Relevant source:

- [`src/store/db/indexes/embeddings_index.go:538`](/home/rowan/Documents/antfly/antfly/src/store/db/indexes/embeddings_index.go#L538)

Those are not obviously conservative "maximize recall" settings. They look like fixed performance-oriented defaults. If recall is poor on this dataset, this config is a very credible place to look first.

### 4. HBC is doing aggressive dynamic pruning during search

The HBC search code prunes internal nodes and leaves based on the current best distance threshold and `Episilon2`:

- internal-node pruning:
  [`lib/vectorindex/hbc.go:1885`](/home/rowan/Documents/antfly/antfly/lib/vectorindex/hbc.go#L1885)
- default distance metric fallback:
  [`lib/vectorindex/hbc.go:288`](/home/rowan/Documents/antfly/antfly/lib/vectorindex/hbc.go#L288)

The specific pruning logic here is aggressive enough that it could absolutely explain a recall ceiling if the heuristic is not well tuned for this workload.

I would treat this as one of the strongest Antfly-side suspects.

### 5. Antfly is searching quantized vectors by default

The embeddings index turns quantization on by default in the HBC config:

- [`src/store/db/indexes/embeddings_index.go:544`](/home/rowan/Documents/antfly/antfly/src/store/db/indexes/embeddings_index.go#L544)

And leaf search uses quantized vectors when available:

- [`lib/vectorindex/hbc.go:1902`](/home/rowan/Documents/antfly/antfly/lib/vectorindex/hbc.go#L1902)

That does not prove quantization is the reason recall is low, but it is another real Antfly-side approximation step in the retrieval path. Between quantization and dynamic pruning, there is enough approximation here that a ~50% recall result is not inherently surprising.

### 6. The running local Antfly binary appears inconsistent with the checked-out source/API expectations

This matters because it raises the possibility that the benchmark is not actually exercising the exact implementation implied by the local source tree.

Observed on the running local Antfly used for the rerun:

- creating the vector index with type `"embeddings"` failed with HTTP 400
- creating it with `"aknn_v0"` succeeded
- inserts using `SyncLevelEmbeddings`/`aknn` semantics for precomputed `_embeddings` failed with a server-side `index not found: vec`
- explicitly including `"name": "vec"` in the index config was necessary for the field-only precomputed-embedding path to work reliably

Those runtime behaviors are notable because the checked-out source clearly presents an embeddings index abstraction:

- the API type includes `EmbeddingsIndexConfig` and documents `distance_metric`:
  [`src/store/db/indexes/openapi.gen.go:170`](/home/rowan/Documents/antfly/antfly/src/store/db/indexes/openapi.gen.go#L170)
- the CLI expects an explicit `name` field in an index definition:
  [`cmd/antfly/cmd/cli/table.go:133`](/home/rowan/Documents/antfly/antfly/cmd/antfly/cmd/cli/table.go#L133)
- the DB tests explicitly expect vector writes to support embeddings-level sync before search:
  [`src/store/db/embeddings_test.go:792`](/home/rowan/Documents/antfly/antfly/src/store/db/embeddings_test.go#L792)
- `SyncLevelEmbeddings` is a first-class sync level:
  [`src/store/db/ops.proto:41`](/home/rowan/Documents/antfly/antfly/src/store/db/ops.proto#L41)

So there is at least some evidence of **runtime/build drift or compatibility mismatch**:

- either the running binary is older or behaviorally different from the checked-out source
- or the field-only precomputed-embedding path is not as solid as the source/tests suggest

Either way, that is an Antfly-side problem, not a benchmark problem.

### 7. The direct query shape itself does not look like the main failure point

The Antfly query path for direct embeddings appears coherent in source:

- request embeddings are copied into the remote index query:
  [`src/metadata/api_query.go:45`](/home/rowan/Documents/antfly/antfly/src/metadata/api_query.go#L45)
- if `indexes` is omitted, Antfly skips query-time embedding generation:
  [`src/metadata/api_query.go:496`](/home/rowan/Documents/antfly/antfly/src/metadata/api_query.go#L496)
- remote search accepts direct embedding searches:
  [`src/store/db/indexes/remoteindex.go:1430`](/home/rowan/Documents/antfly/antfly/src/store/db/indexes/remoteindex.go#L1430)

So my current read is not "the query API is fundamentally broken." The stronger Antfly-side story is that the query reaches the right subsystem, but the subsystem's current implementation and/or runtime configuration is not producing good enough nearest-neighbor quality.

## Most likely Antfly-side explanations, in order

1. There is a cosine/HBC bug in Antfly's indexing path, visible even on the source-built upstream server after the benchmark sends unit-normalized vectors.
2. HBC pruning and search-width defaults are too aggressive for this benchmark.
3. Quantization is costing too much recall on this workload.
4. The running local Antfly binary does not match the behavior implied by the checked-out source, especially around embeddings index creation and sync semantics.
5. There is a field-only precomputed-embedding bug or edge case in the current implementation.

## What I would test next in Antfly

1. Reproduce the source-build cosine panic in a focused Antfly test around `HBCIndex.splitVectorSet()` / `BalancedBinaryKmeans.validateVectors()` with precomputed unit vectors.
2. Verify why Antfly later observes a non-unit vector after the benchmark has already sent `sqnorm=1.0`.
3. Verify the exact binary/version/config that was running for the local ~50% recall rerun.
4. Run the same 50K case with a less approximate Antfly configuration:
   - higher `SearchWidth`
   - lower or disabled pruning
   - quantization disabled
5. Check whether the runtime binary is actually the same code as `~/Documents/antfly/antfly`.
6. Reproduce the `SyncLevelEmbeddings` failure in a focused Antfly test using precomputed `_embeddings`.

## Verdict

The general concern in the thread now looks directionally correct: the remaining low recall appears to be an **Antfly problem**.

The specific root cause is not yet proven to be only "HBC/SPANN pruning," but the evidence now points much more strongly to:

- an Antfly cosine/HBC bug on the source-built path
- Antfly search heuristics/defaults
- Antfly approximation behavior
- and possibly Antfly runtime/build inconsistency

not to simple benchmark miswiring.
