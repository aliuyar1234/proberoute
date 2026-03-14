# Paper Draft Template

## Title
{TITLE}

## Abstract
{ABSTRACT}

## 1. Introduction
- problem setting and motivation
- why adapting pretrained AR LMs to explicit MTP matters
- why frozen-backbone readout adaptation is nontrivial
- project thesis and contributions

## 2. Related Work
- MTP pretraining
- inference-time multi-head adaptation / verification-style decoding
- hidden-state future-signal evidence
- pretrained-model adaptation, dense WHS, and marginalization baselines
- recent objective-side alternatives

## 3. Method
- probe bank and heatmap construction
- probe-initialized sparse routing
- layer-mix variants and baselines
- horizon heads and tied unembedding
- loss and warmup / deephead ablations

## 4. Experimental Setup
- model family and scales
- datasets and deterministic split policy
- token budgets and stopping rules
- evaluation metrics and bootstrap CIs
- finalist-selection rule

## 5. Results
- screening table
- final main comparison table
- probe heatmaps
- acceptance analysis
- confirmatory results if available

## 6. Analysis
- router support patterns
- probe-to-router overlap
- what the results imply mechanistically
- mixed or negative findings if present

## 7. Limitations and Threats to Validity
- model family restriction
- dataset restriction
- acceptance proxy is not a throughput benchmark
- limited seed count and budget
- frozen-backbone scope only

## 8. Conclusion
- what was learned
- whether the main thesis held
- what remains open
