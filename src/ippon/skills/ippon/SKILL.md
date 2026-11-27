---
name: ippon
description: >
  Ask a decision model on this Mac a judgement call and act on its probabilities. It answers yes or no, one of
  named options, or a level on a scale, about text, an image, a voice clip or a video. Use it for a call
  the agent makes again and again (is this safe, is it done, which team, which tool, is this page broken), when
  the call should cost no tokens, stay on the Mac, or take milliseconds. If a call is unsure, hand only that call
  to the LLM. Trigger words are ippon, decision model, classify, route, triage, gate, check, System One and noul.
---

# ippon

ippon runs decision models on this Mac. A decision model reads a state and answers typed
questions about it, with a probability for every allowed answer, and it generates no text. Its answers are the
creator's own, because ippon checks every build against the creator's code.

## When it fits

Use ippon for a judgement call with a fixed set of answers, where the outcome is one of a few labels but choosing it
needs reading. Leave open-ended writing, planning and tool arguments to the LLM. ippon pays most for a judgement
call that the agent makes on every step, file or message, because each answer takes tens of milliseconds and no
tokens.

## The loop

1. Get a build. `ippon recommend --json` (or `recommend images`, `recommend audio`) names the build to start with
   on this Mac as `start_with`, and lists every pick with whether it fits. Pull `start_with`, not the first pick.
   The picks are in order of what was checked of each build, not of which build to start with. Before you download
   it, look for a copy already on this Mac. Look in the Hugging Face cache (`~/.cache/huggingface/hub`, which ippon
   reuses by name), `~/models`, LM Studio's `~/.lmstudio/models` and `ollama list`. If you find a copy, name it by
   its folder or GGUF file path, or as `ollama://TAG`, and ippon loads it in place. If not, `ippon pull MODEL`
   downloads it. The user accepts the model's licence once, so leave that prompt to them. Done when
   `ippon pull MODEL --dry-run --json` reports the build complete, or a local copy answers.
2. Write the request. Give a state and every question about it, each keyed by an id you choose, for example
   `{"state": "...", "questions": {"kind": {"type": "choice", "instructions": "What kind of issue is this?",
   "criteria": {"bug": "Something is broken", "enhancement": "A request for something new"}}, "urgent": {"type":
   "noul", "instructions": "Does it block the user's work?"}}}`. Put a choice's options in `criteria`. If a starter
   request from [Evaluating](https://github.com/sqwishai/ippon/blob/main/docs/evaluating.md#starter-requests)
   matches, start from it. Run `ippon lint request.json --json` until it reports no problem.
   For a graded call (severity, priority, risk), do not ask the model for the grade, because it reads and does not
   reason through a rubric. In the same request, ask small yes/no questions about the facts behind the grade ("Is
   the service down?", "Is the user blocked with no workaround?"), and combine their answers in code. Done when lint
   is clean. For question types, limits, media fields and wording, run `ippon skills ippon requests`.
3. Ask. Ask every question about one state in one request, so the model reads the state once for all of them. For
   one call, run `ippon decide MODEL request.json --json`, and a running server that has the model answers it. For
   many calls, start one shared server (`ippon serve MODEL --detach --json`) and ask it over HTTP, MCP (`decide`) or
   `ippon decide ... --server`. For many states, run `ippon decide MODEL request.json --states states.jsonl` once. It
   asks about every line in one run, and a running server that has the model answers them. Never loop one command
   per state. Give each subagent that shares the server its own name (`IPPON_CLIENT=NAME ippon decide ...`), so
   `ippon status` shows them apart. A `--states` run writes each answer as it comes back, in the states' order, so
   output with fewer lines than the states is a run still going, even when standard error is empty. Wait for the
   command to exit. Done when every answer is back and none says `truncated`. For servers, HTTP and
   errors, run `ippon skills ippon serving`.
4. Act on probabilities. Act on a threshold, not on the top label alone. A noul answer is P(yes), and a choice
   answer gives every option's probability. If an answer falls between your thresholds, the call is unsure, so send
   that call to the LLM or to the user. To choose thresholds from labelled examples, run
   `ippon skills ippon thresholds`.
5. Keep the logic tested. If a call comes out wrong, add the case to `cases.jsonl` with the answer each question
   should have given, for example `{"id": "...", "state": ..., "labels": {"down": true, "blocked": false}}`. After
   every change to the questions, run `ippon eval cases.jsonl --questions request.json --model MODEL` again. Also
   run the cases through your combining code. Done when the change fixes the new case and breaks no old one.

## Facts that change what you do

- Pass local media files by path: over the CLI with `ippon decide MODEL request.json --image PATH --video PATH --audio PATH`
  (each flag repeats), or over MCP by putting local paths directly in `images`, `videos`, or `audio`. Over HTTP, send
  base64 or `data:` URIs, because ippon fetches no URL. Only some builds read each medium. `ippon recommend images`
  and `ippon recommend audio` name the builds whose answers about that medium are checked. GGUF builds (`:q8_0`) read
  images through the encoder file beside them.
- For models that read left to right, a context sent again or grown by one message is cheap, because ippon keeps
  what it has read. Models that read both ways (d1, d3) read a grown or changed context again. An identical request comes from the answer
  cache for every model.
- Errors are JSON with an `exit_code`. If you get `overloaded` (8) or a 503, retry later and never start a second
  server.
