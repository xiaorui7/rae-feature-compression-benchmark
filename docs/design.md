# Design decisions

## Re-extract features with provenance

The benchmark does not repurpose older thesis caches because they did not contain the sample IDs and model/preprocessing hashes required by this protocol. Re-extraction costs encoder time but makes cache identity and split membership auditable.

## Use exact blocked cosine search

The configured galleries fit local memory, so exact search avoids approximate-index recall as an additional variable. Queries are processed in configurable blocks. Gallery IDs are stably sorted before scoring, making exact-score ties deterministic.

## Compare storage formats and compressors

FP16 isolates a storage-format change without reducing nominal dimension. PCA and learned linear adapters add encoder state but reduce per-vector payload. The deployment calculation includes serialized gallery data, encoder-side state, and the train mean instead of reporting only `dimension * scalar size`.

## Keep upstream loading inside extraction measurements

The official Stage-1 constructor loads the decoder even though this benchmark only needs encoded features. Reports disclose that boundary and measured CUDA allocation. The project does not present those numbers as an optimized encoder-only minimum.

## Select on validation data

The recommendation first applies a configured validation Precision@10 tolerance and then selects the smallest measured deployment. Test data is used only for an audit after selection, avoiding test-set tuning.

