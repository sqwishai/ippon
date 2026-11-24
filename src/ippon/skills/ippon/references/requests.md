# Requests

A request holds one state and every question about it, each keyed by an id you choose.

```json
{
  "state": "Order #48213 was charged twice. Please reverse the second charge.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
    "team": {"type": "choice", "instructions": "Which team should handle this message?",
             "criteria": {"billing": "Charges and refunds", "shipping": "Delivery issues", "other": "Anything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is the request?",
                "criteria": ["Can wait", "Needs help today", "Cannot continue working"]}
  }
}
```

## Types and limits

| Type | Asks | Criteria | Answer |
|---|---|---|---|
| `noul` | yes or no | optional `{"true": "...", "false": "..."}` | `noul`, which is P(yes) |
| `choice` | one of its options | labels, or labels to descriptions | `choice`, `confidence`, every label's `probabilities` |
| `score` | a level on 2 or more ordered levels | levels, lowest first | the creator's `score` (the expected level, or the median level for d1), `confidence`, `legend`, `probabilities` |

The model sets how many options or levels a question may have, as many as it has labels or codes for. That is 702
for d1's choices and 10 for its scores, and 255 for d3's. `ippon lint request.json --model MODEL` says when a
request asks for more, and `ippon lint --model MODEL` reports the model's `max_options`.

## State and media

- `state` is text, a JSON object or a JSON list.
- `images` is a list of images. Images may also sit inside the state as `{"type": "image_url", ...}` parts.
- `audio` is a clip or a list of clips (WAV, MP3, M4A, FLAC, Ogg Vorbis, AIFF or CAF), for a model that hears
  (d1-Omni-2B). A question's instructions may themselves be a spoken clip.
- `videos` is a list of videos, for decision models that read video.
- Over HTTP, media is base64 or a `data:` URI, and ippon does not fetch URLs. On the command line, pass local files
  with `--image PATH`, `--video PATH`, or `--audio PATH` on `ippon decide` (or `ippon lint`), and in the MCP `decide`
  and `lint` tools you may pass local file paths directly in `images`, `videos`, or `audio`.
- Only some builds read each medium. `ippon recommend images` and `ippon recommend audio` name the builds whose
  answers about that medium are checked. GGUF builds (`:q8_0`) read images through the encoder file beside them.

## Wording that answers well

- Ask one question for each thing you need to know, as a full question that names what to look at.
- Give a choice options that do not overlap, each with a description of what belongs to it. Add one catch-all
  (`other`, `none`) for a state that fits no other option.
- Ask every question about one state together, never one request per question.
- If you ask one question about many states, keep option descriptions to a few words. Every request reads all of
  them, so long descriptions set the speed more than the states do.
- `ippon lint` flags vague or duplicate instructions, overlapping options, a missing catch-all and options long
  enough to slow every request. With `--model`, it also flags what that model would cut to fit its window.
