# Tzun's reporting style guide (for AI assistants writing reports/summaries on my behalf)

Read this before drafting any report, advisor summary, or write-up for me. Calibrated from my own revision of the proposal-critique report (2026-06). When in doubt, match the reference sample at the bottom, not generic "polished" writing.

## Language

- Reports to my advisor/lab: Traditional Chinese body, with English technical terms kept inline and untranslated (controller, dataset, oracle, regret, framework, brute force, for loop, test set, GMV, fairness, agent, profile, category error, welfare decomposition...). Do not force-translate jargon; if a Chinese gloss helps, put it in full-width parentheses right after, e.g. welfare decomposition（福利分配）.
- Conversation with me: English (unless I start in Mandarin).

## Punctuation rules (hard rules)

- NO em dashes (—). NO semicolons (; ；). Restructure into separate sentences or use commas/colons instead.
- Quotation/emphasis: light single quotes '...' or 「」. I use both; '...' is fine and common in my writing.
- Full-width parentheses（）for Chinese parentheticals, half-width () for English/numbers.
- ~ for approximate numbers (~5 個).
- Inside a multi-line bullet, use `<br>` line breaks rather than nested sub-bullets when laying out enumerated details like (1) (2) (3).

## Structure

- Open with one plain sentence of context, e.g. 我針對...做了一些重新審視，與 AI 討論了...
- Top-level bullets: **bold problem statement**, phrased concretely. It's fine to phrase a problem as the literal question someone would ask, in quotes, e.g. 會被質疑 '爲什麽不要...？'
- Sub-bullets: short factual support, one idea each.
- The fix gets its own bold lead-in: **修改：...** followed by the plan.
- Keep it concise. Cut anything that doesn't change what the reader decides.

## Tone

- Direct, conversational-academic. Not stiff, not salesy.
- Plain-language explanations of technical/econ concepts, often as parentheticals: 乘客的淨收益（這趟車對乘客的價值，扣掉車資和等車的時間成本後，實際賺到多少好處）.
- Honest hedging is good and natural: 應該還是會很昂貴的, 也是有價值的結論, 能的話就很棒. Don't overclaim, don't polish away uncertainty.
- It's OK to keep small colloquial touches. Don't smooth my drafts into formal report-speak.
- Avoid stacked jargon. One technical term per clause where possible, explained on first use if the reader may not know it.

## Reference sample (my own writing, the gold standard for tone)

```markdown
我針對上次提出來的發想做了一些重新審視，與 AI 討論了現階段的 proposal 有哪些漏洞或是不夠完善的地方：

- **問題 1：把 Castillo 的 welfare decomposition （福利分配）當作評估目標是 category error**
  - 他的數據描述的是'Houston 市場下 surge pricing 這個特定政策'的福利分配，不是一個好的 controller 應該達到的目標，而且「重現論文展現的結果」等於重現一個已被證實傷害特定司機族群的結果，與我們的 fairness 動機矛盾。
  - 另外 NYC 計程車資料是管制費率、無價格變化，dataset 的選用需要重新考慮
  - **修改：把驗證箭頭反過來。** Castillo 改為三個角色：
    <br>(1) 動機：說明只看 GMV 會看不到福利重分配
    <br>(2) 用來檢查模擬器合不合理：模擬器的參數（例如需求彈性）需從其他獨立研究取得，完全不用 Castillo 的數字。這樣 Castillo 的結果就可以拿來當 test set。如果模擬器在類似的設定下，能自己跑出他觀察到的現象（乘客受益、司機受損、且集中在工時受限的司機），就代表模擬器的行為是合理的。
    <br>(3) 當作福利報告的範本：學 Castillo 的做法，把'整體福利'拆開來報告，但所有數字都直接從模擬器裡面算出來。例如：乘客的淨收益（這趟車對乘客的價值，扣掉車資和等車的時間成本後，實際賺到多少好處）、司機的淨收益（收入扣掉付出時間的機會成本）、平台利潤（抽成收入），並且可以再細分到不同司機類型（工時受限 vs 彈性）。Controller 的好壞就用這些指標來比較，而不是拿 Castillo 在 Houston 的數字當標準答案。

- **問題 2：Agent 選方法的機制，會被質疑 '爲什麽不要全部方法 for loop 都試一次看哪個最好就可以了？爲什麽要 agent？'**
  - 方法庫只有 ~5 個方法家族 × 3 個子任務，決策空間小到可以 brute force 搜尋。
  - **修改：不主張 agentic pipeline 比暴力搜尋快，而是把 brute force 搜尋説成我們比對的最佳解。** 我們先把數個 (方法組合 × 城市情境) 跑完，得到每個情境的真正最佳解（oracle）。有了標準答案，才能客觀地回答：我們的 framework 只看一個城市的可觀測特徵，能不能預測哪個方法會贏？能的話就很棒，因爲每個城市都要 for loop 跑完很多個 RL 的方法時間成本應該還是會很昂貴的。
    <br>評估方式是：每個城市情境都是一次獨立的執行，我們的 agents 只拿到那個城市的 profile 和方法庫的說明，看不到任何實驗結果，就要選出方法。選完之後，我們才拿預先算好的結果矩陣來對答案，計算它的選擇和真正最佳解（oracle）的差距（regret），並跟 random、固定單一方法、人寫決策表比較。
    <br>選對了，代表 context→method 的對應關係真的可以從資料學到、可以推廣到新城市。選不對，也是有價值的結論。現在一堆論文宣稱 LLM agent 會'自動選方法'，但從來沒有人有標準答案可以檢驗，我們是可以是第一個能給出量化評估的。
```
