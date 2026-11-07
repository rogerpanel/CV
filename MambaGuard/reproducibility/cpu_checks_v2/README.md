# CPU sanity checks of the v2 encoder (not paper results)

Run on a CPU with `--sentence-backend hash`: the MiniLM sentence embedding is
replaced by a hashed bag of words because the sandbox could not download
MiniLM. One seed each. They were used only to (1) check that the pipeline
works and (2) choose the main encoder **on validation macro-F1**
(MCPTox: chunk_mil 0.766 > chunk_lex 0.736). The paper reports the Kaggle GPU
runs with MiniLM.

| folder | what |
|---|---|
| cpu512, cpu2048 | MCPTox, chunk_lex encoder, 512 / 2048 lexical buckets |
| cpumil | MCPTox, chunk_mil encoder |
| cpuho | MCPTox held-out folds (wrapper style, paradigm), chunk_mil + lexical baseline |
| cpuad | AgentDojo, chunk_mil + lexical baseline |
