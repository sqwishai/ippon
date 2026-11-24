# Thresholds

Choose thresholds on the user's own examples, not by guess. A decision model's probabilities are its creators', so
a threshold chosen on examples holds when the model runs elsewhere.

1. Label a few dozen to a few hundred real states, with one JSON line for each state, for example
   `{"id": "t1", "state": "...", "labels": {"team": "billing", "urgent": true}}`.
2. Run `ippon eval labelled.jsonl --questions request.json --model MODEL [--model OTHER] --json`.
3. Read the threshold table for each question. For each confidence from 0 to 0.99, it shows how many answers reach
   it and how many of those are right. For a noul, it also shows the false yeses and noes.
4. Pick the lowest confidence whose accuracy the task can accept.

## System 1, then System 2

Answer locally at or above the threshold, and send the rest to the LLM or the user. Most calls are confident, so
most never leave the Mac, and the unsure ones get the larger model's attention. For a yes/no gate, use 2
thresholds. Act on yes above one and on no below the other, and ask in between.
