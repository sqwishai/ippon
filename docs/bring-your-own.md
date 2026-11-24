# Bring your own decision model

ippon loads your classifier or label-token model in the architectures below, with its published weights and head.
The 2 standard formats need no adapter code. ippon changes nothing about the model unless you ask (`precision="q8"`).
The checkpoints listed at the end were checked against transformers in fp32, and you can check your own model the same
way.

| Your model | What ippon needs | Format |
|---|---|---|
| a `*ForSequenceClassification` checkpoint (BERT, RoBERTa, XLM-RoBERTa, DistilBERT, DeBERTa-v2/v3, ModernBERT, Llama, Mistral, Qwen2, Qwen3, Gemma 3) | nothing, or a `ippon.json` only to template the input | `generic/classifier` |
| a causal LM that answers with a token (yes/no, a label, an end-of-turn token) on Llama, Mistral, Qwen2, Qwen3, Gemma 3/4, Qwen3.5 or LFM2 | a `ippon.json` with the prompt and the answers | `generic/label-tokens` |

Safetensors weights and tokenizer assets are required. ippon never runs code from a model file and never loads weights
from a pickle. A BERT or DistilBERT checkpoint with only its WordPiece `vocab.txt` gets the fast tokenizer that
transformers converts it to.

```python
import json

from ippon import Decider

d = Decider.load("path/or/hub-id")
d.classify("I was charged twice for one order")          # every label's probability
d.classify(["first text", "second text"])                 # read together
d.systemone({"state": "...", "questions": {"q": {"type": "choice", "criteria": ["billing", "sales"]}}})

# a model's ippon.json can be given in code (or `--settings FILE` on the command line) instead of beside its weights
with open("references/byo/settings/Qwen3-Reranker-0.6B.json") as f:
    settings = json.load(f)
r = Decider.load("Qwen/Qwen3-Reranker-0.6B", settings=settings)
```

## Classifiers (`*ForSequenceClassification`)

- ippon runs the checkpoint's own head in fp32, and decoders use `score` at the last token. BERT uses its pooler,
  then `classifier`. RoBERTa and XLM-RoBERTa use `classifier.dense`, tanh, then `out_proj`. DistilBERT uses
  `pre_classifier`, ReLU, then `classifier`. DeBERTa uses `pooler.dense` and its activation, then `classifier`.
  ModernBERT uses its prediction head over [CLS] or the mean (`classifier_pooling`).
- For probabilities, ippon takes a softmax over the labels that a question asks about. For a multi-label model
  (`problem_type: "multi_label_classification"`) or a one-logit model (rerankers, reward models), it takes the
  sigmoid of each label.
- A `choice` question picks among any of the model's labels (`id2label`). A `noul` names a label in its
  `instructions` (or `ippon.json`'s `true_label`), or reads the score of a one-logit model. A `score` uses the labels
  in order as its levels (or `levels`). A multi-label model answers 1 `noul` for each label.
- The model reads the state of the request with the tokenizer's special tokens, as `tokenizer(text)` gives it. A
  `ippon.json` beside the weights templates the input over `{state}` and the question's `{instructions}`. It can also
  give a second text for cross-encoders and NLI models.

```json
{"format": "classifier", "text": "{instructions}", "pair": "{state}"}
```

Questions with the same inputs share one reading. The second text of a BERT pair gets token type 1, as its tokenizer
marks it. ippon refuses an input past the learned positions of a BERT-family model (`ContextTooLong`).

## Label-token models

A causal LM that answers with a token needs a `ippon.json` with the prompt and the answers, as below.

```json
{
  "format": "label_tokens",
  "prompt": "<|im_start|>user\n<Query>: {instructions}\n<Document>: {state}<|im_end|>\n<|im_start|>assistant\n",
  "noul": {"true": "yes", "false": "no"},
  "choice": "{label}",
  "score": "{index}",
  "option": "{label}: {description}",
  "over": "answers"
}
```

Write the prompt out whole, chat markers included, over `{state}`, `{instructions}` and `{options}`. `{options}`
gives each option as `option`, one per line. ippon reads the answer at the last token of the prompt, as the first token
of the answer text of each option. ippon refuses 2 options whose answers start with the same token. Questions about
one state share its reading, and a grown or re-asked state resumes from the kept reading.

`over: "answers"` takes a softmax over those tokens. `over: "vocabulary"` reads the probability of each token among
all of them (an end-of-turn detector's P(end token), where you can then leave out a noul's "false").

## Checked on real checkpoints

Each checkpoint below passed a check against transformers fp32 on the CPU. Tokens were identical, every
probability was within the bound (the larger of 0.01 and transformers' own bf16 distance), and the top answers were
the same. See the [reference generator](../references/byo/make_reference.py) and the
[requests, settings and reference answers](../references/byo/).

| Model | Architecture, head | Largest difference (fp32) | Bound |
|---|---|---:|---:|
| cross-encoder/ms-marco-MiniLM-L6-v2 | BERT, a pair, one logit | 4.8e-07 | 0.01 |
| BAAI/bge-reranker-base | XLM-RoBERTa, a pair, one logit | 8.1e-07 | 0.01 |
| SamLowe/roberta-base-go_emotions | RoBERTa, 28 labels, multi-label | 6.7e-08 | 0.01 |
| MoritzLaurer/ModernBERT-base-zeroshot-v2.0 | ModernBERT, NLI pair, mean pooling | 8.2e-06 | 0.0253 |
| Skywork/Skywork-Reward-V2-Qwen3-0.6B | Qwen3, chat template, one logit | 3.2e-06 | 0.0231 |
| Qwen/Qwen3-Reranker-0.6B | Qwen3 causal LM, yes/no label tokens | 3.1e-06 | 0.0275 |
| ProsusAI/finbert | BERT, 3 labels, vocab.txt only (no tokenizer.json) | 6.6e-07 | 0.01 |
| distilbert/distilbert-base-uncased-finetuned-sst-2-english | DistilBERT, 2 labels, vocab.txt only | 1.2e-07 | 0.01 |
| hfmlsoc/ncii-light-guard-v01 | Gemma 3 270M classifier | 1.7e-06 | 0.0185 |
| unsloth/gemma-3-270m-it | Gemma 3 270M, yes/no label tokens (an 839-token prompt past its 512-token window) | 9.6e-07 | 0.01 |
| protectai/deberta-v3-base-prompt-injection-v2 | DeBERTa-v3, 2 labels (logits within 4.2e-06) | 1.4e-11 | 0.01 |

`ippon verify MODEL --reference references/byo/NAME.json.gz` checks a model on this Mac at any precision.
To get the same check for your own model, write a few requests and run the
[reference generator](../references/byo/make_reference.py) on it.

A model with a custom prompt or head uses an [adapter](adding-a-model.md). Support for an architecture does not make a
checkpoint verified. For example, Gemma 4's layers are checked, but no decision checkpoint on Gemma 4 has a creator
reference yet. [models.md](models.md) records those limits.
