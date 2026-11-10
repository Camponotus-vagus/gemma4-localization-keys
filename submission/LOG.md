# Registro delle sottomissioni

## Piano A/B (dal 25/10)

Una sottomissione ripetuta della stessa configurazione oscilla di circa ±3 task su 58 (SD ~1,6 task, media di 10 ri-sottomissioni di un bundle pubblico). Per misurare l'effetto della skill `repo-localization` si alternano `v1_budget` (controllo) e `v2_skill` (v1 + skill, unica differenza: la skill e il paragrafo del prompt che la richiama), un giorno ciascuno, fino al 10/11: circa 8 sottomissioni per braccio, confronto con test di Mann-Whitney e intervallo bootstrap della differenza delle medie. Nessuna modifica ad altri parametri durante l'A/B.

## Sottomissioni

Una riga per sottomissione. `sha256` = hash del bundle calcolato da `scripts/build_submission_notebook.py`, per risalire alla configurazione esatta in git.

| Data (UTC) | Ref Kaggle | Bundle | sha256 | Modifica rispetto alla precedente | Public LB | Note |
|---|---|---|---|---|---|---|
| 2026-09-28 08:49 | 56635448 | `baseline_v0` | 3c5796a7cd7c | sample ufficiale senza LoRA; budget 5 min, 60 tool call, timeout comando 120 s | 0,06 | ancora in classifica |
| 2026-10-24 09:29 | 57589627 | `v1_budget` | eca6ba42690c | via sub-agent e tool grafo; prompt nostro orientato al budget (prima modifica entro la chiamata 10, diff conta anche con test rosso, igiene del diff, edit via heredoc); 7 min, 30 call, timeout 240 s, max_output 8192 | 0,20 | pattern dei notebook pubblici a 0,17-0,24; rumore ±3 task per sottomissione |
| 2026-10-26 09:08 | 57660961 | `v2_skill` | c3368f341cf4 | A/B arm B: v1_budget + repo-localization skill (only change) | 0,18 | |
| 2026-10-28 01:41 | 57719553 | `v1_budget` | eca6ba42690c | A/B arm A: control | 0,18 | |
| 2026-10-29 10:26 | 57767843 | `v2_skill` | c3368f341cf4 | A/B arm B: v1_budget + repo-localization skill (only change) | 0,13 | |
| 2026-10-31 17:49 | 57846083 | `v1_budget` | eca6ba42690c | A/B arm A: control | 0,15 | |
| 2026-11-02 01:34 | 57890958 | `v2_skill` | c3368f341cf4 | A/B arm B: v1_budget + repo-localization skill (only change) | 0,15 | |
| 2026-11-03 12:33 | 57947667 | `v1_budget` | eca6ba42690c | A/B arm A: control | 0,13 | |
| 2026-11-05 01:11 | 58004691 | `v2_skill` | c3368f341cf4 | A/B arm B: v1_budget + repo-localization skill (only change) | 0,15 | |
| 2026-11-07 18:22 | 58101249 | `v1_budget` | eca6ba42690c | A/B arm A: control | 0,13 | |
| 2026-11-09 07:46 | 58156421 | `v2_skill` | c3368f341cf4 | A/B arm B: v1_budget + repo-localization skill (only change) | 0,17 | |
| 2026-11-10 10:06 | 58196428 | `v1_budget` | eca6ba42690c | A/B arm A: control | in attesa | |
