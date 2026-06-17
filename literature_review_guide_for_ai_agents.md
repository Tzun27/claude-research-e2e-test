# Literature Review Guide for AI Research Agents

> Adapted from the six-step workflow in *Literature Review 怎麼寫才不爛？* (researcher20.com, 2026). The original targets graduate students; this version is rewritten for an AI agent conducting literature review either autonomously or as part of a research pipeline.

---

## 0. Why this guide is different from a general LR tutorial

Most literature review advice assumes a human reader who is bottlenecked on **reading speed and memory**. An AI agent is bottlenecked on different things:

- **Context window**, not reading time. You can ingest 50 abstracts in seconds but cannot hold 50 full papers in working memory.
- **Pattern over-fitting**, not under-reading. You will find spurious "themes" if you let surface lexical similarity drive clustering.
- **Confident summarization**, not laziness. Your default failure mode is producing a fluent paragraph that *sounds* synthesized but is actually a stitched-together summary with hedging language.
- **Source attribution drift**. Without strict bookkeeping, claims get attached to the wrong papers or to no paper at all.

The six-step workflow below is the same as the source article's, but each step is rewritten with these failure modes in mind. **The goal is not to mimic a human researcher's process; it is to produce the same end product (a synthesis-mode literature review) given an agent's actual strengths and weaknesses.**

---

## 1. The core distinction: Summary mode vs. Synthesis mode

This is the single most important idea in the source, and it is the easiest thing for an LLM-based agent to get wrong.

| | Summary mode (bad) | Synthesis mode (good) |
|---|---|---|
| Subject of sentences | Author names ("Chen (2020) found...") | Concepts, claims, debates ("One line of evidence suggests...") |
| Organization | One paragraph per paper | One paragraph per theme/argument |
| Treatment of disagreement | Listed sequentially | Framed as a debate with explanatory hypotheses |
| Author names | Carry the argument | Appear only as citations |
| Reader takeaway | "The agent read N papers" | "Here is the state of the field, and here is the gap" |

**Operational test for an agent**: After drafting any paragraph, check whether the grammatical subjects are mostly author names. If yes, you are in summary mode. Rewrite with concepts as subjects and authors demoted to parenthetical citations.

---

## 2. The six-step workflow

```
Search → Fast-read & tag → Synthesis Matrix → Extract themes → Judge gap → Write with MEAL
```

Each step has a **purpose**, an **agent-specific procedure**, **failure modes**, and an **exit condition** so you know when to move on.

---

### Step 1 — Search and initial filtering

**Purpose.** Build a manageable corpus (target 30–50 papers) without over-filtering.

**Procedure for an agent.**

1. Identify 2–3 recent authoritative review articles or meta-analyses in the target domain. Extract their key concepts, controlled vocabulary, and most-cited references — these become your seed terms and seed papers.
2. Run keyword searches across Scopus, Web of Science, and Google Scholar (or domain-specific databases). Default time window: last 10–15 years, with explicit carve-outs for landmark/canonical works.
3. First-pass filter on title + abstract only. Decision rule: keep if "plausibly relevant," reject only if "clearly off-topic." Do not try to make final inclusion decisions yet.
4. Record for each kept paper: citation key, title, abstract, venue, year, DOI/URL. Persist to a structured store (CSV, JSON, or a reference manager API) — **not** to free text in context.

**Failure modes specific to agents.**

- **Premature filtering by semantic similarity.** Embedding-based filters will discard papers using non-standard terminology (e.g., older terminology, adjacent fields). Prefer keyword + citation-graph expansion over pure embedding similarity.
- **Recency bias.** Search engines and citation services often surface recent work. Manually verify that landmark older works in your seed reviews are in the corpus.
- **Single-source bias.** If you only use Google Scholar, you inherit its ranking biases. Cross-check across at least two databases.

**Exit condition.** A list of 30–50 papers with metadata, no full-text reading yet. If you have <20, your search is too narrow. If you have >100, tighten keywords rather than reading all of them.

---

### Step 2 — Fast read and tag

**Purpose.** Build a *retrieval index*, not a writing draft. After this step, you should be able to ask "which papers discuss measurement of X?" and get an answer in one query.

**Procedure for an agent.**

1. For each paper, extract a fixed schema. Do not produce free-form summaries:
   - `population` (sample / subjects)
   - `method` (experimental / observational / meta-analytic / qualitative / theoretical)
   - `key_finding` (one sentence, ≤25 words, structured as "X → Y under condition Z" if possible)
   - `effect_size` (with units, or "n/a")
   - `limitations` (3–5 keywords, not prose)
   - `stance` (which theory/camp it supports, contests, or is neutral toward)
   - `tags` (controlled vocabulary you maintain across the corpus)
2. Use a controlled tag vocabulary. Maintain a tag dictionary; do not invent new tags freely or you will fragment retrieval. Reconcile near-duplicates (`#cognitive_load` vs `#cog_load`) at the end of each batch.
3. Persist to a structured store (Zotero with tags, a SQLite DB, a Notion/Airtable database, or a JSON file). The store is your external memory — do not rely on context.

**Time/budget guidance.** The source suggests 15–20 minutes per paper for a human. For an agent, the analogous discipline is: **read abstract → introduction's last paragraph (research aims) → conclusion → results section only if needed to fill the schema.** Do not read full method sections at this stage.

**Failure modes specific to agents.**

- **Hallucinated effect sizes or sample sizes.** If a number is not in the abstract or conclusion, mark `n/a` rather than guessing from context. Verify against the paper before any number enters a downstream artifact.
- **Tag explosion.** Without a controlled vocabulary, every paper gets unique tags and the index becomes useless. Cap tags at ~30 across the corpus and merge aggressively.
- **Fluent-but-wrong stance attribution.** Papers often cite a theory without endorsing it. Distinguish "uses theory X as framework" from "supports theory X" from "tests theory X."

**Exit condition.** Every paper in the corpus has a complete schema entry. Tag vocabulary is reconciled. You can answer retrieval queries by tag without re-reading.

---

### Step 3 — Build a Synthesis Matrix

**Purpose.** Force a **lateral** view of the corpus. A summary indexes by paper; a matrix indexes by concept across papers. This is the structural pivot from summary mode to synthesis mode.

**Structure.** A table where rows are papers and columns are dimensions of comparison. Minimum columns:

| Author (Year) | Population | Core finding (X → Y relationship) | Method limitations | Relevance to current study |
|---|---|---|---|---|

You will typically build **multiple matrices**, one per sub-question or construct, not one giant matrix for the whole field.

**Procedure for an agent.**

1. Decide the **comparison axis** before populating. The axis is a question like "How does construct X relate to outcome Y?" or "What measurement instruments are used for Z?" — not "everything about the topic."
2. Populate from your Step 2 schema entries, not by re-reading. If a cell requires information you did not extract, either go back to Step 2 for that paper or mark `unknown`.
3. Sort rows by the dimension that reveals the most variation (often year, population type, or effect direction). Sorting is part of the analysis — different sorts surface different patterns.
4. Generate the matrix as a structured artifact (CSV, Markdown table, or spreadsheet). Do not embed it only in prose.

**Failure modes specific to agents.**

- **Inventing comparability.** If two papers measured X with non-comparable instruments, do not put their effect sizes in the same column without flagging. Add a `comparability_notes` column when in doubt.
- **Filling cells to look complete.** Empty cells are informative — they often indicate a gap. Resist the urge to write something in every cell.
- **One giant matrix.** If your matrix has 50 rows and 12 columns, no pattern will be visible. Split it.

**Exit condition.** One or more matrices, each focused on a single comparison axis, with all cells either populated from extracted data or explicitly marked `unknown`.

---

### Step 4 — Extract themes from the matrix

**Purpose.** Convert the matrix into a small number of *thematic claims* about the field. The source identifies three theme types:

1. **Consensus zone** — claims supported across multiple papers, methods, and populations. These become your "background" paragraphs.
2. **Controversy zone** — points where findings diverge or interpretations conflict. These are where you demonstrate critical reading.
3. **Gap zone** — important questions that the corpus has not adequately addressed. These set up your research question (or, for an agent doing a survey, the open problems section).

**Procedure for an agent.**

1. For each matrix, scan for patterns along three dimensions:
   - **Effect direction agreement.** Do most papers find the same sign? Same magnitude?
   - **Moderator structure.** Do disagreements track with population, method, or context?
   - **Coverage holes.** Are there populations, methods, or conditions that the matrix does not contain rows for?
2. Draft thematic claims using this template: *"On [specific question], [pattern] holds, [except/moderated by] [condition], [and] [open problem]."*
3. Verify each thematic claim against the matrix: list the rows that support it, the rows that contradict it, and the rows that are neutral. If you cannot do this listing, the theme is not yet evidence-grounded.

**Example transformation (from the source).**

- Bad theme statement: *"A, B, and C all studied cognitive load."*
- Good theme statement: *"Two camps disagree on cognitive load measurement: objective physiological indicators vs. subjective scales, and the two methods produce frequently inconsistent results."*

The good version names the construct, names the disagreement, and points to the empirical pattern (inconsistency across methods).

**Failure modes specific to agents.**

- **Lexical clustering masquerading as thematic clustering.** Embedding-based or keyword-based clusters group papers by *vocabulary*, not by *argument*. A theme is an argumentative position, not a topic label.
- **Over-smoothing.** LLMs tend to write "the literature suggests..." even when the literature is contradictory. If a claim has counter-evidence in the matrix, the theme statement must acknowledge it.
- **Too many themes.** A literature review typically has 3–6 themes, not 15. If you have more, you are still describing topics rather than identifying argumentative positions.

**Exit condition.** A short list of themes (consensus / controversy / gap), each backed by an explicit list of supporting and contradicting papers from the matrix.

---

### Step 5 — Judge whether a gap is a real research gap

**Purpose.** Distinguish "no one has done X" from "X is a research gap worth filling." This is the step where most weak literature reviews fail, and the source is emphatic about the criteria.

**The three criteria.** A real gap must satisfy *all three*:

1. **Importance.** Does filling this gap meaningfully change our understanding of the construct, theory, or phenomenon? Or is it a parameter sweep?
2. **Reasonableness.** Does the gap connect to a theoretical or practical question that the field cares about? Can you state *why* someone should fill it?
3. **Feasibility.** Can the gap be addressed with available methods, data, and resources?

**Bad gap statement (from the source).**

> "Past research mostly used Western samples; this study uses a Taiwanese sample."

This says only "no one has done it." It does not say why a Taiwanese sample would change the answer.

**Good gap statement (from the source, paraphrased).**

> "The applicability of cognitive load theory to East Asian educational contexts remains contested: Chen (2020) and Wang (2021) report effect sizes only 30–40% of those in Western studies in Taiwanese samples, but it is unclear whether the difference reflects cultural variation, measurement misfit, or features of the educational system. The current study compares a locally adapted scale against the original to determine the boundary conditions of the theory's cross-cultural applicability."

The good version specifies the gap, explains why it matters theoretically, and indicates how the proposed work addresses it.

**Procedure for an agent.**

1. For each candidate gap from Step 4, write three sentences — one per criterion — answering the criterion explicitly.
2. If you cannot write a non-trivial sentence for any criterion, the gap is not yet a real gap. Either find supporting evidence or drop it.
3. **Specifically check importance.** "Population X has not been studied" is rarely sufficient on its own. Ask: is there theoretical reason to expect different results in population X? If yes, that reason is the actual gap, not the population label.

**Failure modes specific to agents.**

- **"No one has done" framing.** The agent's training data includes many papers that motivate themselves this way, so the language comes naturally. It is still wrong.
- **Manufacturing importance.** Resist the temptation to inflate importance with general phrases ("this has implications for..."). State the specific theoretical or practical consequence.
- **Ignoring feasibility.** A gap that requires data you cannot get or methods that do not yet exist is not a gap your project can fill. Be honest about scope.

**Exit condition.** Each gap statement passes all three criteria with explicit justification. Gaps that fail are dropped or rewritten.

---

### Step 6 — Write paragraphs using the MEAL structure

**Purpose.** Translate themes and gaps into prose where every paragraph has a function. MEAL ensures that each paragraph makes a claim, supports it, interprets it, and connects it forward.

**The MEAL elements.**

| Element | Function | Example phrasing |
|---|---|---|
| **M**ain idea | The paragraph's central claim, in one sentence at the top | "The literature presents two contrasting views on the relationship between X and Y." |
| **E**vidence | Evidence drawn from the literature, framed by position not by author | "One line of work, conducted in laboratory settings, finds positive effects (Chen, 2020; Lin, 2021); a contrasting line, conducted in field settings, finds null associations (Wang, 2022)." |
| **A**nalysis | An explanation of *why* the evidence patterns the way it does | "This divergence likely reflects methodological differences: standardized lab tasks differ from the complex, multi-stimulus environment of real classrooms." |
| **L**ink | A connection to the current study's design, question, or scope | "Because the present study uses a quasi-experimental design closer to Wang's (2022) field setting, we expect attenuated effects relative to Chen (2020)." |

**Procedure for an agent.**

1. Draft one MEAL paragraph per theme from Step 4.
2. For the Evidence sentence, **make the grammatical subject a position or a finding, not an author**. Author names should appear in parentheses or as discourse-light citations, not as the entity doing the action.
3. For the Analysis sentence, propose a mechanism, scope condition, or methodological reason for the pattern. If no analysis is possible, the theme may not be substantive enough.
4. For the Link sentence, state a specific design implication or hypothesis for the current work. Vague phrases like "this informs our research" do not count.
5. After drafting, run the **summary-mode test**: highlight the grammatical subjects of each sentence. If most are author names, rewrite.

**Failure modes specific to agents.**

- **MEAL as decoration.** Producing four sentences in the right shape but with the same summary-mode content. The structure is necessary but not sufficient.
- **Weak Analysis.** Defaulting to "more research is needed" or "results are mixed." These are non-explanations. Give a *reason* the results vary.
- **Generic Link.** "This is relevant to the present study" is not a link. The link must be specific to a design choice, hypothesis, or scope decision.

**Exit condition.** Every paragraph in the literature review section has an identifiable M, E, A, and L. Author names function as citations, not as sentence subjects. Each paragraph either establishes consensus, frames a controversy, or sets up the gap.

---

## 3. Before / After example (reproduced from the source)

### Before — summary stitching (anti-pattern)

> Chen (2020) studied the effect of gamified learning on undergraduate mathematics achievement and found that students using a gamified platform scored significantly higher than those in traditional instruction (d = 0.45). Wang (2021) examined the role of gamification elements on learning motivation and found that points and leaderboards boosted short-term motivation but had no significant long-term effect. Lin (2022) focused on gamification in programming education and found significant improvements in students' problem-solving ability.

The reader learns what each paper said but not what the *field* says, and not how the cited works connect to the writer's own study.

### After — thematic synthesis (target pattern)

> The evidence on gamified learning shows clear context dependence. In controlled experimental settings, positive effects are commonly reported: gamified groups outperform conventional instruction in undergraduate mathematics (Chen, 2020, d = 0.45) and demonstrate gains in problem-solving in programming education (Lin, 2022).
>
> Whether these effects persist in long-term, real-classroom settings is less clear. A longitudinal study found that points and leaderboards raised motivation initially but the experimental–control gap disappeared at three months (Wang, 2021). This "short-term gain, long-term decay" pattern aligns with self-determination theory's prediction that extrinsic rewards can undermine the durability of intrinsic motivation (Deci & Ryan, 1985).
>
> Taken together, short-term gains from gamified learning are reasonably well supported, but the mechanisms of long-term maintenance and the boundary conditions across subject areas remain unclear. The current study uses a quasi-experimental design with a one-semester follow-up to test whether gamification effects persist in authentic classroom contexts.

What changed:

| Dimension | Before | After |
|---|---|---|
| Organizing logic | By author | By theme (consensus → controversy → gap) |
| Relations among sources | Independent | In dialogue, forming an argument |
| Critical content | None (description only) | Yes (identifies contradictions, invokes theory) |
| Connection to current study | None | Explicit — justifies design choice |

---

## 4. Summary checklist for an agent

Before declaring a literature review section done, verify each item:

- [ ] **Corpus**: 30–50 papers, drawn from at least two databases, with seed papers from review articles.
- [ ] **Schema completeness**: every paper has a structured entry (population, method, finding, effect size, limitations, stance, tags) in an external store.
- [ ] **Tag hygiene**: controlled vocabulary, no near-duplicates, no orphan tags.
- [ ] **Synthesis matrices**: one or more, each with a single comparison axis, with `unknown` cells explicitly marked rather than fabricated.
- [ ] **Themes**: 3–6, each labeled as consensus / controversy / gap, each backed by an explicit list of supporting and contradicting papers.
- [ ] **Gap justification**: every claimed gap passes the importance / reasonableness / feasibility test in writing.
- [ ] **MEAL paragraphs**: every paragraph has M, E, A, L. Grammatical subjects are concepts and positions, not author names.
- [ ] **Summary-mode audit**: re-read the draft and confirm that no paragraph is structured as "Author X said... Author Y said... Author Z said..."
- [ ] **Forward link**: the literature review connects, by the end, to a specific research question, hypothesis, or design choice.

---

## 5. The single most useful starting point

The source recommends: if you are stuck, **start at Step 3** by populating a Synthesis Matrix from papers you have already read. The act of building the matrix forces the shift from summary mode to synthesis mode, because the matrix has no row for "everything paper X said" — it only has cells for specific dimensions of comparison.

For an AI agent, this advice translates directly: when the literature review draft is going poorly, do not rewrite the prose. Go back and build (or rebuild) the matrix. Prose quality follows from matrix quality.
