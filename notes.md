## Experiment 1 — Baseline Prompt (v1)

### Objective
Evaluate whether Gemini can reliably classify Spotify app reviews into predefined issue categories and assign appropriate severity scores.

### Experimental Setup
- **Model:** gemini-3.5-flash-lite
- **Prompt version:** v1
- **Evaluation dataset:** 50 manually labeled reviews
- **Classification:** Primary theme, secondary theme, sentiment, severity (0–5), and explanation
- **Storage:** SQLite (`results.db`)
- **Evaluation:** `evaluate.py`

### Results

| Metric | Original labels | Corrected labels |
|---|---:|---:|
| Exact theme agreement | 48% | 84% |
| Lenient theme agreement | 58% | 92% |
| Cohen's kappa | 0.43 | 0.81 |
| Exact severity agreement | 52% | 58% |
| Severity within ±1 | 90% | 90% |
| Mean absolute severity error | 0.60 | 0.54 |

### Label Corrections
The initial human annotations contained inconsistencies with the classification codebook.

18 of the 50 reviews were corrected, including:
- Merging `crash_performance` and `playback_bug` into `app_bug`.
- Reclassifying positive reviews without complaints as `praise_no_issue`.
- Distinguishing feature requests from general praise.
- Clarifying issues involving offline downloads and malfunctioning playback controls.

The original labels are preserved in `labeling_set.csv`, and the corrected labels are stored in `labeling_set_v2.csv`. All corrections are documented in `labeling_set_v2_changes.csv`.

**Methodological limitation:** The corrected labels were developed during the initial evaluation process, so the 84% agreement should be treated as a development-set result, not an unbiased estimate of generalization performance.

### Error Analysis
After label correction, 8 primary-theme mismatches remained.

Key sources of disagreement:
1. Reviews describing multiple issues without a clearly dominant complaint.
2. Confusion between playback bugs, offline downloads, and free-tier restrictions.
3. Feature requests appearing within otherwise positive reviews.
4. Ambiguous UI complaints.
5. Disappearing saved playlists versus disappearing downloaded content.

### Conclusion
Prompt v1 achieved 84% exact theme agreement and Cohen's kappa of 0.81 against the corrected labels. Severity predictions were within one point of human judgments for 90% of reviews.

The results provide a promising development baseline, but further validation is required.

### Planned Improvements for v2
- Improve primary-theme selection for multi-issue reviews.
- Clarify distinctions between overlapping categories.
- Better recognize actionable complaints or feature requests embedded within positive reviews.
- Preserve the same 50-review evaluation set for development comparison.
- Subsequently evaluate the selected configuration on a separate, independently labeled test set.

### Next Experiment
**Prompt v2 — Refined Category Definitions and Decision Rules**

Status: Not yet evaluated.