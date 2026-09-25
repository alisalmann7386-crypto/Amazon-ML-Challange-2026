# Validation and error analysis

1. Establish candidate micro recall: retrieved true links / all true links. Also inspect per-S1 recall and candidate-count tails. Increasing k trades compute for recall.
2. Split by S1 before evaluating matching decisions. Pair-level random splits leak the same business across training and validation. The implementation additionally connects S1 records sharing a labeled S2/S3 target.
3. Train a fresh scaler and classifier in each group fold. Fit unsupervised retrieval on the available target catalog; report this transductive text-only assumption explicitly.
4. Produce OOF pair probabilities. Tune the threshold on the **full S1 set**, retaining entities with no retrieved candidates. Positives lost during retrieval remain false negatives.
5. Reserve an untouched group holdout for final evaluation; the maximum OOF threshold score is a tuning result. Near-duplicate S1 records not linked by supplied labels may still require manual group construction.
6. Run leave-one-country-out experiments to test transfer, then evaluate France only when labels legitimately become available. Do not optimize on test labels or external business lookup.
7. Analyze singleton false merges, shared business names at different addresses, missing addresses, numeric conflicts, abbreviations, transliteration and true multi-match entities.

The CLI `evaluate` command accepts a held-out directory using train filenames and its ground truth, plus already generated outputs. For prediction, expose the same held-out source files under test filenames in a separate directory; never put held-out labels into the training directory.

The metric's empty-truth behavior is explicit. A singleton with any prediction receives zero. Missing or extra S1 rows fail validation rather than silently disappearing from the mean.
