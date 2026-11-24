# Choosing a model and checking it on your own examples

To evaluate a model, measure the probabilities it gives on your own examples. ippon runs each model with its published
weights, prompts and readouts. You choose the model and build, the words of your questions, and what you do with the
answers.

## Pick a model

Run `ippon recommend` to see the build to start with for each model family on this Mac.

```sh
ippon recommend              # text: routing, classification, moderation, extraction checks
ippon recommend images       # models that read images; `audio` for speech
```

For each build, `recommend` shows its weights and what is checked of it against its creator's code
([accuracy.md](accuracy.md)). It also shows whether the build fits this Mac's memory as published. The order is fixed.
Verified builds come first, then the builds whose prompts are checked. In each group, the builds that fit come first,
then the smallest by recorded weight size, from the [recorded model checks](models.md).

`recommend` does not tell you about speed or about which model answers your task best. To find that,
[measure it on your examples](#measure-it-on-your-examples).

ippon never swaps a build that does not fit for a smaller one, because a smaller build or 8-bit weights
(`kettei-core:q8`) change the model's answers. `recommend` names the smaller builds, and you make the choice.

## Download it and see a first answer

Run `ippon pull` to download a model and see its answer to a short sample request.

```sh
ippon pull kettei-core
```

`pull` tells you which build a name chose and why, its size, and whether it fits. Then it downloads the build and
answers a short sample request with it. It shows the probability of each option and the time the answer took. ippon
chooses the kernels for this Mac's chip when the model loads ([kernels.md](kernels.md)). The kernels keep the weights
as published. Their floating-point rounding can differ from MLX's own matmuls, and [accuracy checks](accuracy.md)
measure those differences.

After the first download, a model loads again from the Hub cache at the commit it was downloaded at. It does not ask
the Hub, so it also loads with `HF_HUB_OFFLINE=1` or with no network. A name such as `kettei-core` loads the commit
its answers were checked at, so `ippon pull NAME` again takes no later change ([models.md](models.md)). A Hub id with
no revision (`org/name`) takes the newest commit each time you pull it.

Each answer tells you which build produced it. In Python, the `Result` that ippon's own `Decider.decide` returns has
it as `result.diagnostics["build"]`. A server reports it
through `?diagnostics=1` and `GET /v1/models`.

## Write the questions

Start from one of the [starter requests](#starter-requests) below. Put your own state in it and change its questions
for your task. Then run `ippon lint` to find what to fix.

```sh
ippon lint request.json                # what to fix in it, without a model
ippon lint request.json --model kettei-core  # and as that model reads it: tokens, what would be cut
```

A decision model answers best when each question asks one thing, in full, and names what to look at. The options of a
choice must not overlap. Give each option a description of what belongs to it. Add a catch-all option (`other`,
`none`) for a state that fits none of them. `lint` flags a request that breaks these rules.

`lint` asks for a catch-all only when a choice has 3 or more options, because a choice of 2 is often a plain
either-or. It also flags options whose descriptions run past about 4,000 characters. Every request reads all the
descriptions, so a question asked about thousands of states answers faster with a few words per option. With a model,
`lint` also flags what that model cannot be asked, such as more options than it takes, and shows what ippon would cut
to fit the model's window. When `lint` finds a problem, it exits with 1, so you can run it in CI.

## Measure it on your examples

To measure a model, label real states and run `ippon eval` on them. Label a few dozen to a few hundred real states,
one JSON object per line.

```json
{"id": "t1", "state": "I was charged twice for order #48213.", "labels": {"team": "billing", "urgent": true}}
{"id": "t2", "state": "Where is my parcel? It was due Monday.", "labels": {"team": "shipping", "urgent": false}}
```

Each row carries its `questions`, or all rows share the `questions` of one file (`--questions request.json`). The
label of a noul is true or false (or "yes"/"no"). The label of a choice is the label of its option. The label of a
score is the index of its level (0 the lowest) or the text of that level. Then run the evaluation.

```sh
ippon eval labelled.jsonl --questions request.json --model kettei-core --model kettei-dot
```

Before you compare models, pull each one (`ippon pull d3-mini`). If ippon cannot load a model, the report shows it as
`not loaded`, and the other models still answer. The command then exits 1, because the comparison is not the one you
asked for.

ippon loads the models one at a time and closes each one before the next loads. It answers every row in batches, as
`decide_many` answers them. For each model and question, the report gives these results.

- accuracy and macro-F1 (the mean of the F1 of each option, so a rare option counts as much as a common one)
- the confusion matrix, expected (rows) by answered (columns)
- refused rows (a request the model cannot be asked) and truncated rows (answered with input cut to fit)
- a threshold table. For each confidence (the probability of the top option) from 0 to 0.99, it gives how many
  answers reach it and their accuracy. For a noul, it also gives the false yeses and noes among them. The table shows
  what you get when you keep only confident answers and send the rest to a person. The answers themselves do not
  change.
- the flagging of a yes/no question. A row counts as flagged when the model's P(true) is at least a threshold t, as a
  judge flags a failing trace. The report gives precision, recall and F1 at 0.5 and from 0.1 to 0.9. It gives the t
  with the highest F1 that meets `--min-precision` and `--min-recall`. It checks that choice on held-out rows
  (`--folds`, 5 by default). Each fold is flagged at the t chosen on the other folds, and the report counts the folds
  with no t that meets the floors.

  The report also gives the calibration of P(true), which is its Brier score and, by tenths, the mean predicted
  against the share that were true. Rows that belong together, such as the steps of one agent trace, can carry a
  `"group"`. The rows of a group stay in one fold, so ippon never chooses a threshold on rows of the trace that it then
  judges.

Set the floors your use needs first. Trust the held-out line more than the chosen line, because the chosen line was
picked on the same rows that it is judged on. 0.5 is often the wrong threshold for a judge. A model that gives every
failure 0.40 and every pass 0.10 catches every failure at 0.30 and none at 0.50. Hamel Husain and Antaripa Saha show
this in [How to Use Decision Models for Evals](https://hamel.dev/blog/posts/decision-model-evals/).

```sh
ippon eval traces.jsonl --model kettei-core --min-precision 0.8 --min-recall 0.8
```

For two or more models, the report then lists every row where their answers differ. ippon writes the answers of every
row to a results file (`labelled.results.jsonl`, or `--out`), with the probability of every option as the model gave
it. `--json` prints the report as JSON.

## Answer here, send the unsure ones on

Add `unsure_below` to a request to find the answers that need a person or a larger model. The response lists under
`unsure` each question whose most likely option has a probability under that threshold. Act on the other answers
immediately. A local model settles most questions on its own, and only the few that it is unsure of are worth a person
or a larger model.

```sh
ippon decide kettei-core:q8 '{"state": "I was charged twice.", "questions": {...}, "unsure_below": 0.8}'
```

```python
from ippon import Decider

with Decider.load("kettei-core:q8", accept_license=True) as model:
    result = model.decide(ticket, questions)
for qid, answer in result.answers.items():
    if answer.confidence < 0.8:  # the rest are settled here
        ask_a_larger_model(ticket, qid)
```

The same field works over HTTP and in the MCP `decide` tool. At a terminal, ippon marks an unsure answer `(unsure)`.
Choose the threshold from your own examples. For each confidence, `ippon eval` shows how many answers reach it and how
many of those are correct ([Measure it on your examples](#measure-it-on-your-examples)).

## Run it on many states

Use `--states` to ask the questions of one request about many states.

```sh
ippon decide kettei-core:q8 request.json --states states.jsonl > answers.jsonl
```

ippon asks each state in `states.jsonl` the questions in `request.json` and answers them in batches. Each line of
`states.jsonl` is a JSON string, or an object with `state` and optionally `id`, `images`, `audio`. ippon writes one
JSON line out for each line in. Each output line is the response, or the error of that line alone.

## Starter requests

These 4 requests do common jobs, and they ask their questions in the way that decision models read best.

**ticket-routing** routes a support ticket to the team that should handle it and says how urgent it is.

```json
{
  "state": "Hi, I was charged twice for order #48213 and the second charge still hasn't been refunded. Please reverse it before my card statement closes on Friday.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this support ticket?",
      "criteria": {
        "billing": "Charges, refunds, invoices, payment methods or subscriptions.",
        "shipping": "Delivery, tracking, returns or items that arrived damaged.",
        "account": "Signing in, passwords, profile details or closing the account.",
        "technical": "The product not working as it should: errors, crashes or bugs.",
        "other": "Anything none of the other teams covers."
      }
    },
    "urgent": {
      "type": "noul",
      "instructions": "Does the customer need an answer today, because money, a deadline or their safety is at stake?"
    },
    "mood": {
      "type": "score",
      "instructions": "How upset is the customer in this ticket?",
      "criteria": [
        "calm",
        "annoyed",
        "angry"
      ]
    }
  }
}
```

**classification** sorts a piece of product feedback by its kind and its sentiment.

```json
{
  "state": "The new export button is great, but the CSV it writes loses the date column.",
  "questions": {
    "kind": {
      "type": "choice",
      "instructions": "What kind of feedback is this message mainly?",
      "criteria": {
        "bug": "Something in the product doesn't work as described.",
        "feature-request": "A wish for something the product doesn't do yet.",
        "praise": "Only approval of something that works well.",
        "question": "Asks how to do something, with no complaint or wish.",
        "other": "Feedback that is none of these."
      }
    },
    "sentiment": {
      "type": "score",
      "instructions": "What is the overall sentiment of this feedback?",
      "criteria": [
        "negative",
        "mixed",
        "positive"
      ]
    }
  }
}
```

**moderation** checks a user's message against a community's rules.

```json
{
  "state": "Anyone who still uses this library is an idiot. Buy my course instead: cheap-courses.example",
  "questions": {
    "rule": {
      "type": "choice",
      "instructions": "Which of the community's rules does this message break most clearly?",
      "criteria": {
        "none": "It breaks none of the rules.",
        "harassment": "Insults, threats or demeaning remarks aimed at a person or a group.",
        "spam": "Advertising, self-promotion or links unrelated to the discussion.",
        "sexual": "Sexual content of any kind.",
        "self-harm": "Encourages or describes ways of harming oneself.",
        "other": "It breaks a rule not listed here."
      }
    },
    "remove": {
      "type": "noul",
      "instructions": "Should a moderator remove this message under the community's rules?"
    }
  }
}
```

**extraction-check** checks that the fields extracted from a document match what the document says. The due date in
the example is wrong on purpose.

```json
{
  "state": {
    "document": "INVOICE INV-2041. Issued 2026-10-01 by Northwind Ltd to Contoso GmbH. Total due: 1,250.00 EUR, payable by 2026-11-30.",
    "extracted": {
      "invoice_number": "INV-2041",
      "total": "1,250.00 EUR",
      "due_date": "2026-10-30"
    }
  },
  "questions": {
    "invoice_number": {
      "type": "noul",
      "instructions": "Does the document give exactly the invoice number in extracted.invoice_number?"
    },
    "total": {
      "type": "noul",
      "instructions": "Does the document give exactly the total amount and currency in extracted.total?"
    },
    "due_date": {
      "type": "noul",
      "instructions": "Does the document give exactly the due date in extracted.due_date?"
    }
  }
}
```
