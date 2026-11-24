# Adding a model

A model family plugs in through an adapter. An adapter is a small Python class that names the files it needs,
recognises a download, and loads it into a backbone and a format. The runtime decides how to run the prompts, and the
adapter never does.

`ippon new-adapter creator/model --arch qwen35` writes the package that this page describes, and marks the parts to fill
in with TODO. The snippets below are an outline of an adapter and do not make a complete package that runs. The example
readout is yes/no only, so validate unsupported question types before rendering.

## The adapter

An adapter subclasses `AdapterBase`, as in this example.

```python
from ippon import API_VERSION, LoadContext, Loaded, ModelDir
from ippon.adapter import AdapterBase
from ippon.arch import qwen35


class Acme(AdapterBase):                      # the defaults of the optional members (adapter.py)
    api = API_VERSION
    name = "acme/decider"                     # as the entry point is named
    files = ("config.json", "acme.json", "tokenizer.json", "*.safetensors", "model.safetensors.index.json")

    def matches(self, model: ModelDir) -> bool:
        return model.json("acme.json") is not None

    def load(self, model: ModelDir, ctx: LoadContext) -> Loaded:
        backbone = ctx.backbone(qwen35, model)    # in the load's dtype, quantized to its bits
        return Loaded(backbone, AcmeFormat(ctx.tokenizer(model), backbone.output_matrix))
```

- For the layers, reuse an `ippon.arch` family when the model is a fine-tune of one, as most models are. The families
  are `qwen35` (`causal=False` for attention that reads both ways), `lfm2` (causal or bidirectional), `lfm3` (Liquid's
  d1 models, which `adapters/d1.py` loads), `dense` (Llama, Mistral, Qwen2, Qwen3), `gemma3`, `gemma4`, `modernbert`,
  `bert` and `deberta`. `arch.DECODERS` and `arch.ENCODERS` name the family for the model_type of a config.json, for
  every family but `lfm3`. Load the family with
  `ctx.backbone(module, model, **its_options)`, which passes the dtype and quantization of the load. If a head is
  stored beside the backbone, use
  `ctx.weights(model, exclude=("head.safetensors",))`.
- A new family implements `arch.base.Backbone` (`prefill`, `branch`, `batch`, `new_cache`, `output_matrix`, `causal`),
  usually by giving `arch.common.Passes` its `_forward`. It lists its model_types in its own `FAMILIES` and joins the
  decoder or encoder modules in `arch/__init__.py`. Test it against an independent implementation, as
  `tests/test_qwen35.py` and `tests/test_backbone_contract.py` do.
- For the heads, `readout` has the common ones. These are label tokens (`LabelSet`, `label_logits`,
  `option_probabilities`) and a learned row per option code (`CodeReadout`). A head that reads every position asks
  for every row in the readouts of its prompt.
- For a LoRA on a public base, name the base in the adapter's `base(model)`, as (repo, revision). ippon downloads the
  weights of the base at that revision before it checks that the model fits, so the size of the base counts. ippon then
  gives the base to `load` as `ctx.base`. Merge the adapter into it
  (`ctx.weights(ctx.base).with_lora(adapter_weights, adapter_config)`) and pass the merged weights to `ctx.backbone`. If quantization is asked for, it applies after the
  merge. `adapters/kev.py` does all of this.
- For the files, the Hub download patterns are restricted to `.safetensors`, `.json`, `.txt`, `.model`, `.jinja` and
  `.md` assets and `LICENSE` (`hub.ALLOWED`). ippon never imports Python files from the model repository. If the adapter
  needs a pickle, it names the pickle exactly in its `read_safely`, and `ptfile` reads it without running anything from
  it (`adapters/kev.py`).
- If the model has GGUF builds for llama.cpp, the adapter declares in `gguf_rules` how their tensors map to the
  creator's names. These rules are `gguf.names.Reads`, built from the architecture tables in `gguf/names.py`. The
  adapter also checks the metadata of each build against the creator's configuration in `gguf_check`
  (`adapters/d1.py`).

## The format

A format renders a request into prompts and reads the scores of each question, as in this example.

```python
from ippon import Rendering
from ippon.adapter import FormatBase
from ippon.plan import Prompt, Readout, state_end_marks
from ippon.readout import LabelSet, label_logits, option_probabilities


class AcmeFormat(FormatBase):
    def __init__(self, tokenizer, output):
        self.tokenizer, self.output = tokenizer, output
        self.labels = LabelSet(((tokenizer.encode(" Yes")[0],), (tokenizer.encode(" No")[0],)))

    def render(self, request):
        texts = [f"State: {request.state}\nQuestion: {q.instructions}\nAnswer:" for q in request.questions]
        end = len("State: ") + len(str(request.state))  # the state's end: a later request resumes there
        prompts = [                          # one prompt per question: the state first, so it is shared
            Prompt(tuple(ids), (Readout(q.id, len(ids) - 1),), state_end_marks(text, offsets, end))
            for q, text, (ids, offsets) in zip(request.questions, texts, self.tokenizer.encode_offsets(texts))
        ]
        return Rendering(request, tuple(prompts))

    def scores(self, rendering, hidden):     # per question: its probabilities in question.options order
        return {
            q.id: option_probabilities(label_logits(hidden.rows(q.id)[-1:], self.output, self.labels.flat), self.labels)[0]
            for q in rendering.request.questions
        }
```

Follow the creator's code exactly. This includes the strings, the order in which it shows criteria, how it handles a
missing state or description, its limits and its temperature.

A prompt's `marks` show where the part that a later request will likely send again ends, which is usually the state.
The decider keeps the prefix up to that mark, so an agent that asks about the same state again does not read it again.
Mark a position only where the tokens of every request split there, such as the end of a state that the creator
tokenizes on its own. For a state that grows at its end (an agent's conversation, text or JSON),
`plan.state_end_marks` gives the position from which it resumes, a few tokens before its end. `plan.one_mark` falls
back to the end of a state that is cut short.

Keep 1 mark per prompt, because every mark costs each recurrent layer a split of its pass and a copy of its state.
`answer()` (the creator's response shape) and `input_tokens()` (its usage count) have defaults. If the creator's
server reports something else, override them.

What several creators share has a single definition to reuse. `schema.systemone_answer` shapes an answer with the creator's
confidence measure, which is `formulas.above_chance` and `formulas.near_best`, the System One format's choice
and score confidences, or your own. `formulas.floored` commits a weak yes/no to a floor, and `schema.rounded` rounds the numbers
of an answer. Both shape only the served answer, and the probabilities that `answer` returns stay the model's, which
`verify` compares. `adapters.qwen.chat` writes the chat template of a Qwen3.5 checkpoint, with thinking off.
`readout.answer_tokens` finds the token that each answer letter or code is read as after the prompt, and refuses a
tokenizer that splits one.

A model that reads images or audio declares this in `media`. Use `frozenset({"images"})`, "audio" for a clip with the
state, or "spoken" for questions whose instructions are a clip (`Question.spoken`). The decider refuses media to a
format that does not declare them. The request's `images` arrive as `media.Media` (bytes, sha256, format), checked
where they entered. `media.load_rgb` decodes one as the creator's code does.

Each row of a medium goes into the prompt as a placeholder token, where the creator puts the medium. The token is
`arch.common.media_id(digest, row)`, with the first 48 bits of the medium's sha256. `encode_media(renderings)` then
returns the rows `[rows, hidden]` of each medium by that digest, and the backbone reads them in place of the
placeholders. The format of d1 (`adapters/d1.py`) and its towers (`media/lfm3.py`, `media/lfm3_audio.py`) are the
example. The rows of its images open the user turn, where its chat template puts them.

## Registration

```toml
# the adapter package's pyproject.toml
[project.entry-points."ippon.adapters"]
"acme/decider" = "acme_ippon:Acme"
```

After `pip install acme-ippon`, `Decider.load("acme/decider-1b")` finds the adapter. In a script or notebook,
`ippon.register(Acme())` does the same.

## The checks

1. Make a reference with `references/<adapter>/make_reference.py`. Run it in the creator's own environment, never
   inside ippon. It runs the creator's code on `references/requests/standard.json`, in fp32. It writes the token ids
   of every prompt, the probabilities of every question in the request's option order, and the creator's token count.
   See `references/d1/make_reference.py`.
2. Make sure that `ippon verify <model> --reference <file>` passes at every precision offered. The tokens must be
   identical, the probabilities within the bound, and the top answers the same outside near ties
   ([accuracy.md](accuracy.md)).
3. Run `ippon bench <model>` on at least 1 Mac, against the other runtimes of the model
   ([benchmarks.md](benchmarks.md)).

Add the model to the README's short-name/creator/input table, and put its architecture and licence in models.md.
Record the model and each build in `src/ippon/verified.py`, with how each one came out. Record a GGUF build with its
pinned file, a build past its bound with its distance, and the model with its place in setup's list. From this file,
`uv run python scripts/models_table.py docs/models.md` writes the table in [models.md](models.md), and
`ippon pull --dry-run` reads it.
