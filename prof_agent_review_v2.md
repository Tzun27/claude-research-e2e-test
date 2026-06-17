# Senior Reviewer's Assessment of `literature_review_ride_hailing_ml_2018_2026_v2.md`

*Reviewer: senior academic referee, posture: skeptical-but-fair. Reviewed against the actual papers in the same directory.*

---

## 1. Overall assessment

The review is at **major-revision** status — close to defensible as a thesis chapter or survey but not yet ready. Its strongest virtues are structural: it follows a disciplined six-step workflow, exposes its own evidence in matrices, and (commendably) uses a v2 changelog to publicly correct prior extraction errors. Its decisive weaknesses are that the author flagged three load-bearing numbers as "unverified" while the PDFs sat in the same directory and could have been verified in minutes (two of the three numbers do not survive verification); that one method's framework name (`FOVIR` for Yang 2024) appears to be fabricated — the paper's actual term is `ODCMG`; and that the corpus is too narrow (n=27 against the author's own 30–50 target) and too concentrated on Didi/Lyft/NYC to support the field-level theme claims it makes. The synthesis prose is genuinely synthesis, not annotated bibliography, but several themes are anchored on single papers, which is unsafe.

---

## 2. Major issues, ordered by severity

### (A) The "(unverified)" label is procedurally indefensible — and two of the three numbers are wrong

Section 0 of the review tells the reader that Enders 2023 (5–17% profit), Hoppe 2024 (2–6% performance), and Zhang 2023 (+17% revenue / −14% vehicles) "remain unverified because the present review pass could not retrieve the source PDF cleanly." All three PDFs are present in the directory the review is being written from. I retrieved them and read the relevant sections:

- **Enders 2023**: the paper reports "up to 5% over the 20 test dates" on average and "up to 17%" on individual test dates. The review's framing as a "5–17% profit gain" range is a misreading — these are an average and a peak from different statistical objects, not bounds of a range. The schema entry should be rewritten and the matrix row corrected.
- **Hoppe 2024**: COMA^scl beats LRA by up to 1.9% and beats greedy on average by 3.5%, with up to 6% on single test dates. There is no "2–6% performance lift attributable specifically to reward globalization" — the gains are not isolated to the globalization mechanism, and the 2–6% range conflates baselines.
- **Zhang 2023**: verified (+17% revenue and −14% vehicles are the maxima; averages are 6.4% / 10.6%, which the review should also report).

This is the failure mode the author warned about in Section 0 (fluent prose laundering an upstream extraction error). The author's response — demote to "illustrative status" — is not a fix; it is a documentation of the failure. A revision must (i) actually open the PDFs, (ii) re-extract the numbers, and (iii) re-derive whatever theme prose depends on them. As of v2, the Theme 1 anchor papers (Wang 2025, Hu 2025) substitute for Enders/Hoppe, but Theme 4's controversy claim still rests on Zhang 2023 + Castillo 2025, an n=2 base.

### (B) Castillo 2025 numbers verify but the citation requires audit

The +3.53% total welfare, +6.97% rider, −1.97% driver figures are correct in direction and magnitude; the v1→v2 inversion correction is a real fix. However, the PDF in the directory appears to be a **December 2019 working-paper version**, not the Econometrica 2025 published version. The published article exists (Econometrica vol. 93, 2025, "Who Benefits from Surge Pricing?"), but the review should cite the paper version it actually read, and verify whether welfare numbers in the published version match the working-paper numbers — counterfactual estimates often shift between drafts. Listing the source as "Econometrica 93 2025" while reading a 2019 working paper is a citation-hygiene problem.

### (C) Yang 2024's framework name is wrong throughout

The review uses "FOVIR" — in Section 2.1, Matrix 2 row "Yang2024 (GRC)", and Theme 3 — to describe the cooperative Markov game formulation. The paper's actual term is **ODCMG (Order-Dispatching Cooperative Markov Game)**, introduced in Section 2.2 of the paper. "FOVIR" does not appear in the paper. This is exactly the kind of confident-sounding-but-wrong technical token that AI-generated reviews are known to hallucinate, and the v1→v2 changelog process did not catch it. Search-and-replace fix is straightforward; what is more important is to re-check the rest of the schema for similar invented acronyms.

### (D) Tresca 2026 numbers do not align with the paper

Spot-checks against the paper's tables produced three discrepancies:

- "Within 2.3% of oracle (NYC)" — Table I (NY 16-station) shows −4.2% from oracle. The −2.3% figure is closer to the Luxembourg mesoscopic afternoon-peak row (~−2.4%) in Table V. The location attribution is wrong.
- "1.5% loss across fidelity" — the macro→meso zero-shot gap in the paper is on the order of 60%, not 1.5%. I cannot find a 1.5% figure in any table that matches the claim as stated.
- "18% gain via meta-RL on disturbance" — meta-RL vs. single-city baseline on the Brooklyn special-event scenario is roughly a 61% gain (39.8 vs. 24.7); meta-RL vs. fine-tune is essentially a tie. The 18% is not where the review puts it.

This paper is the corpus's only "scale + transfer" anchor for the AMoD bi-level line; if its numbers are wrong, Theme 5 (gap C, cross-city transfer) and Matrix 3's "scale + transfer" cell both need to be re-derived. The review must either re-read the paper or remove the specific magnitudes.

### (E) GARLIC's empty-load delta is internally inconsistent

The review reports "empty-loaded rate 38.23% (vs. 40.71% best baseline); 0.30 km error" alongside "−7.6 pp empty-loaded rate vs offline RL." Forty point seven one minus thirty-eight point two three is **2.48 pp**, not 7.6 pp. Either (i) the −7.6 pp figure is against a different (much weaker) baseline, in which case the review must name that baseline and report the corresponding rate, or (ii) the figure is incorrectly transcribed. As written, the two sentences contradict each other arithmetically. This is a Theme 6 anchor and Matrix 5 cell; both need correction.

### (F) Corpus size is below the author's own threshold and too geographically concentrated to support the theme claims

The review acknowledges 27 papers vs. a 30–50 target and acknowledges Didi/Lyft/NYC concentration in Section 1.1, but several of the substantive theme claims are unsafe at this corpus size:

- Theme 4 ("DRL pricing in this corpus does not engage with OR/economics") is anchored on **a single DRL pricing paper** (Zhang 2023). One paper cannot characterize a literature; the controversy framing is overstated even after the v2 hedge ("within the corpus reviewed"). Either expand the corpus or rewrite Theme 4 as "this corpus does not contain a DRL pricing paper that engages with OR/economics — see §11 for what should be added." The current framing implies more than the evidence supports.
- Theme 6 ("LLM role unstandardized") is anchored on **four LLM papers**. The author's own counter-reason in §7.2 (premature consolidation risk) is correct and should be moved up into the theme statement.
- Theme 1 (RL-proposed weights + combinatorial matcher) is the safest claim, but the case that the template is "no longer uncontested" rests on Yue 2024 (verified, 0.7–3.9% TDI) and Lyu 2025 (preprint, peer review pending) — a thin contestation.

A reviewer of a thesis chapter would tell the candidate to either re-scope the question (e.g., "RL-based dispatching in the Didi/Lyft production lineage, 2018–2026," for which the corpus is adequate) or expand to the §11 list before defending field-level claims.

### (G) Suspected coverage gaps the author should justify or fix

The §1.1 bounded-out list is honest, but several omissions still warrant flags. I mark these as **suspected** rather than confirmed where I have not independently surveyed the relevant subfield:

- **Mean-field-game RL** — the author admits this was "bounded out partly in error in v1." It should be in. As bounded out, Theme 4's controversy claim is not just overstated; it is wrong in direction (the bridge already exists and the field has been building it for ~5 years). Suspected omissions: Guo et al. (2019) on MFG-RL, Cui & Koeppl (2021) on approximately-stationary MFG-RL, the Yang Yaodong line.
- **Deep-RL surge/dynamic-pricing literature** beyond Zhang 2023 — Liu, Wu, et al.; Mao et al.; Tang et al.'s Didi-side work on pricing/incentive RL. *Suspected.* If even two more DRL pricing papers exist with strategic-driver assumptions, Theme 4's framing collapses.
- **Tang et al.'s Didi MARL line** (KDD 2019, "A Deep Value-network Based Approach for Multi-Driver Order Dispatching") — likely a Thread A foundational citation that bridges Xu 2018 to Eshkevari 2022. The review jumps from Xu 2018 to Eshkevari 2022 with no intermediate.
- **Alonso-Mora et al. 2017 and capacity-constrained pooling** — bounded out, but the review treats Hu 2025 BMG-Q as the pooling anchor; without the OR-side pooling literature, the comparison Hu 2025 invites is unmoored.
- **Driver-labor empirics** (Hall–Horton–Knoepfle, Cook et al.) — bounded out, but Theme 4's "adopt strategic agents as a structural assumption" recommendation in Section 8 needs *empirical* backing for the strategic-driver assumption, not just the analytical OR papers. Without it the recommendation is methodologically circular.

### (H) Critical engagement with methodology is uneven

The review is self-critical of its own extraction errors but mostly accepting of the cited papers' methodology. Specific gaps:

- **Switchback experimentation** (Azagirre 2024 / Lyft) is described as giving "bounded but coarse causal attribution" — switchbacks have well-documented identification issues under spatial spillovers and time-varying treatment effects (Bojinov & Shephard; Glynn–Johari–Rasouli). Production deployment numbers should be qualified explicitly.
- **OPE variance** (Yang 2024) is correctly flagged at the per-paper level, but the corpus-wide pattern — that OPE estimates drive several of the headline numbers (Yang 2024 5–6%, Yue 2024 0.7–3.9%) — is not synthesized. A theme on "what we actually know vs. what OPE tells us" would strengthen the review.
- **Publication bias in deployed-RL reports** is unaddressed. Production deployments that fail are not published; the corpus reads the deployed-system papers as evidence of RL's general benefit at scale, but every single one of those papers is the platform's own success report.
- **Confounding with concurrent non-RL improvements** — Xu 2018 explicitly mentions that the ~10% completion-rate gain came from a driver-select→platform-assign change, *not* from RL. The review notes this in the schema but does not draw the corpus-wide point that production-deployment percentages confound RL with everything else shipped at the same time.

### (I) The gap analysis under-delivers

§7 is the strongest section the v2 added (counter-reasons per gap, verdicts that turn on which side is stronger). But the gaps it ends up endorsing are general ("three-way joint optimization," "shared LLM benchmark," "cross-city transfer"). These are not specific enough to motivate a research question — they are research areas. A thesis chapter needs at least one falsifiable, scoped, methodologically tractable researchable claim. The closest the review comes is in §8: "benchmark a DRL three-way controller against Castillo 2025's welfare decomposition on Houston data." That is a real research question; it should be promoted into §7 as the load-bearing gap.

---

## 3. Minor issues

The "27 papers" claim in §1.1 disagrees with the threadwise sum (Thread A: 6, B: 8, C: 4, D: 4, E: 4 + Sun2024 cross-referenced = 26 unique + 1 cross-listing = 27 if Sun2024 is counted in both threads or only once). The accounting is correct but should be made explicit. The tag-dictionary axis counts in §2.6 are off by small amounts in several axes ("five listed" with six items, "ten listed" with ten items); the bracket commentary partly acknowledges this but the dictionary should be cleaned. "Foundation model" vs. "ST foundation model" overlap is real and should be either merged or strictly partitioned. The "Wiley/Econometrica" database listing in §1.1 is unusual phrasing — Econometrica is published by Wiley but is a single-journal target, not a Wiley search. The "Tresca 2026 (in press, venue confirmation pending)" hedge is honest but should be rechecked before a final submission. "Sun2024 (JDRCL) — TKDE 2024 (KDD'22 ext.; venue confirmation pending)" — same concern; the review should commit to verified venues. The phrase "within the corpus reviewed" appears so many times that it functions as a verbal tic; readers will not need it on every paragraph if §1.1 is robust. §10's audit checklist is self-attested with no external verification; replace with a `verified_against_pdf` column in the schema that explicitly lists which numbers were re-read from source.

---

## 4. Suggestions for revision, in priority order

1. **Open the three "unverified" PDFs** and rewrite the schema entries (Enders 2023: 5% average / 17% peak, not a range; Hoppe 2024: revise to the COMA^scl-vs-LRA / vs-greedy decomposition; Zhang 2023: report the average alongside the maximum). Re-derive Theme 4 from the corrected Zhang 2023 numbers.
2. **Fix the FOVIR→ODCMG error globally** (Sections 2.1, Matrix 2, Theme 3, anywhere else it appears) and audit the schema for other invented acronyms.
3. **Re-read Tresca 2026** against tables I, V, and VI; either correct or remove the 2.3% / 1.5% / 18% claims.
4. **Resolve the GARLIC arithmetic inconsistency** by identifying which baseline the −7.6 pp gap is computed against and reporting that baseline name.
5. **Audit the Castillo 2025 citation** (working paper vs. Econometrica published version) and confirm the welfare numbers in the version cited.
6. **Expand the corpus by 6–10 papers** from the §11 list — at minimum the MFG-RL line (which is the bridge Theme 4 currently denies) and one or two additional DRL-pricing papers — before resubmitting Themes 4 and 6.
7. **Promote the Castillo-benchmark research question** from §8 into §7 as the load-bearing gap.
8. **Add a short subsection on methodological caveats common across the corpus** (switchback identification, OPE variance, publication bias, concurrent-improvement confounding); right now these critiques are scattered or absent.
9. **In §10 replace self-attested checkmarks with a `verified_against_pdf` column** listing each headline number's verification source.

---

## 5. What the review does well (preserve in revision)

The six-step workflow is real, not cosmetic — Sections 2 (schema), 3 (matrices), 5 (themes), 7 (gaps) build on each other and are not interchangeable. The MEAL prose in §5 is genuinely synthesis: the grammatical subjects are concepts and findings, not authors, and topic sentences carry weight rather than naming people. The v2 changelog in the header is a model of academic self-correction (the Castillo direction inversion is a serious error and is owned cleanly). Matrix 2's lateral observation that "no paper uses the rider as the agent" is the single most insightful corpus-specific finding in the document and deserves more than the one paragraph it gets. The §1.1 bounded-out list with defenses, and the §11 expansion plan with seed citations, are exactly what a senior reader expects to see; do not let them get trimmed in revision. The honest acknowledgment that the corpus is below the guide's threshold is preferable to an inflated claim. Section 8's operational implications make the review useful to a downstream researcher in a way most surveys are not.

The path forward is real-revision-not-cosmetic: open the PDFs that were available all along, fix the wrong numbers, fix the wrong acronym, expand the corpus to where the field-level claims become safe, and the revised document will be defensible.

---

## Appendix: Verification log (numbers cross-checked against the source PDFs)

| Claim in review | Paper | Verdict | Source-of-truth |
|---|---|---|---|
| Castillo 2025: +3.53% welfare, +6.97% rider, −1.97% driver, Houston | `Castillo2025_Surge_Welfare_Econometrica.pdf` | **Verified** (numbers and direction); citation hygiene flag (PDF is a 2019 working-paper version) | Abstract; Section 1 |
| Yang 2024: ~5–6% IORR/IGMV (City B OPE) | `Yang2024_Rethinking_Order_Dispatching_KDD.pdf` | **Verified** as numbers; **framework name `FOVIR` is fabricated** — paper says `ODCMG` | Table 1, Section 2.2 |
| Enders 2023: 5–17% profit | `Enders2023_Hybrid_MADRL_AMoD_L4DC.pdf` | **Partially contradicted**: 5% average / 17% peak are different objects, not a range | Section 5 |
| Hoppe 2024: 2–6% from reward globalization | `Hoppe2024_GlobalRewards_MADRL_AMoD_L4DC.pdf` | **Contradicted**: 1.9% over LRA; 3.5% avg over greedy; up to 6% on single dates; not isolated to globalization | Table 1, Section 4.1 |
| Zhang 2023: +17% revenue, −14% vehicles | `Zhang2023_FutureAware_Pricing_Matching_AAAI.pdf` | **Verified** (these are maxima; averages 6.4% / 10.6%) | Abstract; Results |
| Tresca 2026: 2.3% from oracle (NYC) | `Tresca2026_RoboTaxi_Scale_TCNS.pdf` | **Contradicted/misattributed**: Table I NY shows −4.2%; closest match is Luxembourg meso (~−2.4%) | Tables I, V |
| Tresca 2026: 1.5% loss across fidelity | same | **Not found**: macro→meso zero-shot gap is much larger (~60%) | Section VI-D |
| Tresca 2026: 18% meta-RL gain | same | **Misreported**: meta-RL vs single-city ≈ 61% gain | Section VI-D |
| GARLIC: 38.23% vs 40.71%, −7.6 pp | `Han2025_GARLIC_AAAI.pdf` | **Internally inconsistent**: 40.71 − 38.23 = 2.48 pp, not 7.6 pp | Table 1 |
| Xu 2018: 0.5–5% revenue | `Xu2018_LargeScale_OrderDispatch_Didi_KDD.pdf` | **Verified** | Abstract |
| Eshkevari 2022: 1.3% / 5.3% | `Eshkevari2022_RLinTheWild_Didi_KDD.pdf` | **Verified** | Abstract / experiments |
| Yue 2024: 0.7–3.9% TDI / 1–2% CR | `Yue2024_End2End_Dispatch_Didi_CIKM.pdf` | **Verified** | Table 4 |
| Zhang 2024 NondBREM: 3.76% ORR | `Zhang2024_NondBREM_OfflineRL_AAAI.pdf` | **Verified** | Table 2 |
| Azagirre 2024 (Lyft): ≥$30M/yr | `Azagirre2024_Lyft_RL_Match.pdf` | **Verified** | Abstract |
| Gammelli 2021: 1.6–2.2% from oracle, NY/Chengdu, 4×4, zero-shot | `Gammelli2021_GNN_RL_AMoD_CDC.pdf` | **Verified** | Tables I–II |
| Gammelli 2023: 86.7%–99.5% of oracle | `Gammelli2023_GraphRL_BiLevel_ICML.pdf` | **Verified** (range supported across Tables 1–3) | Tables 1–3 |
| Singhal 2024: 89% oracle, 100× speedup | `Singhal2024_Electric_AMoD_GraphRL_ECC.pdf` | **Partially verified**: 89% at NY 5-region; broader range 55–89% across scales | Table I, Figure 3 |
| UniST: MAE 26.84 zero-shot, +10.1% | `Yuan2024_UniST_KDD.pdf` | **Verified** | Table 4, results |
| UrbanGPT: MAE 6.16 NYC, ~+28% | `Li2024_UrbanGPT_KDD.pdf` | **Verified** | Table 1 |
| LLM-ODDR: +5.21–48.87% GMV; 1–5 s | `Lyu2025_LLM-ODDR.pdf` | **Verified** | Table I; Section V.E |
| JDRCL: TKDE 2024, max-min fairness, convergence | `Sun2024_JDRCL_TKDE.pdf` | **Verified** | Theorem 2; Appendix A |

*Verification is one referee's pass; deeper audit of magnitudes (especially Tresca 2026 and GARLIC) is recommended before resubmission.*
