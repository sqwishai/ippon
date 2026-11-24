"""Deterministic benchmark requests, the same for every engine (stdlib only, so every venv can import it).

A context is a support ticket with its agent trace (customer messages, agent replies, tool calls with JSON
results, log excerpts, internal notes), built from seeded slot-filled templates, so it reads as real text
rather than repeated filler. Its size is a target in tokens, estimated as CHARS_PER_TOKEN characters a token
and cut at a word boundary (calibrated on the d1 (LFM2) and d3 (Qwen3.5) tokenizers: within about 10% of the
target on each; every adapter also records the engine's own token count).

Every timed request gets a unique reference ID at the very start of its state, so no prefix cache can hit
across requests (only the chat template before the state can be shared). The re-sent scenarios are the exception,
as an agent re-sends its context at each step: `repeat` asks one state new questions each time, `grow` adds a
customer message to it before each request. `lines` numbers the lines of one context, L1 to Ln, and asks one yes/no
question about each line in one request: a question per line with the whole document in view, so the number of
questions grows while the document stays the same size.

    python gen_requests.py --show 300          # print one context
    python gen_requests.py --dump requests.json
"""

from __future__ import annotations

import argparse
import json
import random

SIZES = (300, 1000, 2000, 4000, 8000, 16000)
QUESTION_COUNTS = (1, 3, 5)
SCENARIOS = ("fresh", "repeat", "grow", "lines")
GROW = 60  # tokens a grown state gains before each request: a customer message, about one agent turn
# Characters a token of this text (IDs, JSON and timestamps make it about 1.6 tokens a word).
CHARS_PER_TOKEN = 3.7

# ---------------------------------------------------------------------------------------------- questions

QUESTIONS = [
    {"id": "refund", "type": "noul", "instructions": "Is the customer asking for a refund or a credit?"},
    {
        "id": "team",
        "type": "choice",
        "instructions": "Which team should handle this ticket next?",
        "criteria": {
            "billing": "Charges, refunds, invoices and plan changes",
            "technical": "App faults, errors, outages and integrations",
            "shipping": "Deliveries, returns and damaged goods",
        },
    },
    {
        "id": "urgency",
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["Can wait for the normal queue", "Needs a reply today", "Blocking the customer right now"],
    },
    {"id": "churn", "type": "noul", "instructions": "Is the customer threatening to cancel or leave?"},
    {
        "id": "tone",
        "type": "choice",
        "instructions": "What is the customer's overall tone in the latest message?",
        "criteria": {
            "positive": "Satisfied or thankful",
            "neutral": "Matter-of-fact",
            "negative": "Frustrated, angry or disappointed",
        },
    },
    # 6 to 16: for requests with many questions (the first five, and so the requests above, are unchanged)
    {"id": "promised", "type": "noul", "instructions": "Has an agent already promised a fix, a refund or a callback?"},
    {
        "id": "topic",
        "type": "choice",
        "instructions": "What is the main topic of this ticket?",
        "criteria": {
            "billing": "Charges, invoices or plans",
            "access": "Logging in or account access",
            "delivery": "Orders, tracking or returns",
            "defect": "Something broken or not working",
            "other": "Anything else",
        },
    },
    {
        "id": "satisfaction",
        "type": "score",
        "instructions": "How satisfied does the customer sound?",
        "criteria": ["Very unhappy", "Unhappy", "Neutral", "Satisfied", "Very satisfied"],
    },
    {"id": "reference", "type": "noul", "instructions": "Does the ticket mention a specific order or invoice number?"},
    {
        "id": "channel",
        "type": "choice",
        "instructions": "Which channel should the next reply use?",
        "criteria": {"email": "A written reply by email", "phone": "A call back", "chat": "The live chat"},
    },
    {"id": "credited", "type": "noul", "instructions": "Has a refund or credit already been issued?"},
    {
        "id": "complexity",
        "type": "score",
        "instructions": "How complex is this ticket to resolve?",
        "criteria": ["One reply settles it", "Needs another team", "Needs an engineering fix"],
    },
    {
        "id": "area",
        "type": "choice",
        "instructions": "Which product area is affected?",
        "criteria": {
            "payments": "Charges and payment methods",
            "app": "The app or the dashboard",
            "shipping": "Physical deliveries",
            "account": "Users, seats and permissions",
        },
    },
    {"id": "deadline", "type": "noul", "instructions": "Does the customer mention a deadline?"},
    {"id": "error", "type": "noul", "instructions": "Is there an error message or a log excerpt in the thread?"},
    {
        "id": "next_step",
        "type": "choice",
        "instructions": "What should the agent do next?",
        "criteria": {
            "reply": "Answer the customer",
            "escalate": "Hand it to a specialist",
            "refund": "Issue a refund or credit",
            "close": "Close the ticket",
        },
    },
]


# Extra independent ticket decisions for the 50-question workload. Keep the first 16 unchanged so
# historical results reproduce their original prompts. Different IDs never stand in for repeated questions.
QUESTIONS += [
    {"id": name, "type": "noul", "instructions": instructions}
    for name, instructions in (
        ("duplicate_charge", "Does the customer report being charged more than once for one purchase?"),
        ("late_delivery", "Does the ticket report a delivery arriving later than promised?"),
        ("damaged_item", "Does the customer report a damaged or defective physical item?"),
        ("login_failure", "Is the customer unable to sign in to their account?"),
        ("workaround", "Has someone provided a temporary workaround?"),
        ("escalated", "Has this ticket already been handed to a specialist team?"),
        ("replacement", "Has a replacement item been sent or requested?"),
        ("cancel_order", "Does the customer want to cancel an order?"),
        ("cancel_plan", "Does the customer want to cancel a subscription?"),
        ("invoice_correction", "Does the customer request a corrected invoice?"),
        ("multiple_users", "Does the problem affect more than one user?"),
        ("data_export", "Does the ticket concern an incomplete or failed data export?"),
        ("account_security", "Does the ticket raise a concern about account security?"),
        ("price_change", "Does the customer question a price increase?"),
        ("repeat_contact", "Has the customer contacted support about this problem before?"),
        ("follow_up", "Is an agent waiting for more information from the customer?"),
        ("resolved", "Does the latest message say the problem has been resolved?"),
        ("outage", "Does the ticket mention a service outage or incident?"),
    )
]
QUESTIONS += [
    {"id": name, "type": "choice", "instructions": instructions, "criteria": criteria}
    for name, instructions, criteria in (
        (
            "owner",
            "Who needs to act next?",
            {"customer": "The customer", "support": "Support", "specialist": "A specialist"},
        ),
        (
            "evidence",
            "What evidence is most useful here?",
            {"logs": "Service logs", "payment": "Payment records", "tracking": "Delivery tracking"},
        ),
        (
            "scope",
            "How broad is the reported problem?",
            {"single": "One user", "team": "A team", "service": "A whole service"},
        ),
        (
            "remedy",
            "What remedy is the customer seeking?",
            {"refund": "Money back", "fix": "A technical fix", "information": "An explanation"},
        ),
        (
            "stage",
            "What stage is this case at?",
            {"new": "First report", "investigating": "Being investigated", "resolved": "Resolved"},
        ),
        (
            "payment_state",
            "What is the relevant payment status?",
            {"pending": "Pending", "captured": "Captured", "refunded": "Refunded"},
        ),
        (
            "contact_reason",
            "Why did the customer contact support?",
            {"failure": "Something failed", "change": "They want a change", "question": "They need information"},
        ),
        (
            "reply_style",
            "What should the next reply emphasize?",
            {"apology": "An apology", "instructions": "Steps to follow", "confirmation": "Confirmation of an action"},
        ),
    )
]
QUESTIONS += [
    {"id": name, "type": "score", "instructions": instructions, "criteria": criteria}
    for name, instructions, criteria in (
        (
            "business_impact",
            "How much does this affect the customer's work?",
            ["Minor inconvenience", "Slows work", "Stops work"],
        ),
        ("frustration", "How frustrated does the customer sound?", ["Calm", "Concerned", "Very frustrated"]),
        (
            "resolution_progress",
            "How far has the case progressed?",
            ["No action", "Action in progress", "Resolution confirmed"],
        ),
        ("clarity", "How clear is the customer's request?", ["Unclear", "Partly clear", "Clear"]),
        (
            "financial_impact",
            "How serious is the reported financial impact?",
            ["None stated", "One charge", "Repeated or substantial loss"],
        ),
        ("effort", "How much further investigation seems needed?", ["Little", "Some", "Extensive"]),
        (
            "retention_risk",
            "How likely does the customer sound to leave?",
            ["No indication", "Dissatisfied", "Explicitly intends to leave"],
        ),
        (
            "evidence_quality",
            "How specific is the evidence in the ticket?",
            ["General complaint", "Specific symptoms", "Records or logs"],
        ),
    )
]


def questions(n: int, first: int = 0) -> list[dict]:
    """Return n distinct mixed questions from the first-th on, wrapping round (`repeat` turns the list one place
    each request); reject counts the generator cannot actually supply."""
    if not 1 <= n <= len(QUESTIONS):
        raise ValueError(f"question count must be 1..{len(QUESTIONS)}, got {n}")
    return [QUESTIONS[(first + i) % len(QUESTIONS)] for i in range(n)]


def line_questions(n: int) -> list[dict]:
    """One yes/no question about each of a `lines` state's n numbered lines."""
    return [
        {"id": f"L{i}", "type": "noul", "instructions": f"Does line L{i} explain why the customer is unhappy?"}
        for i in range(1, n + 1)
    ]


def asked(req: dict) -> list[dict]:
    """The questions a planned request asks: one per numbered line (`lines`), else `questions`."""
    return line_questions(req["nq"]) if req.get("lines") else questions(req["nq"], req.get("first", 0))


def numbered(text: str, n: int) -> str:
    """`text` cut into n lines of about equal words, each led by its number (L1: ...)."""
    words = text.split()
    cuts = [round(i * len(words) / n) for i in range(n + 1)]
    return "\n".join(f"L{i + 1}: {' '.join(words[cuts[i] : cuts[i + 1]])}" for i in range(n))


def to_systemone(state: str, qs: list[dict]) -> dict:
    """The System One schema (d1, d3, llama.cpp /v1/systemone): {state, questions: {name: question}}."""
    out = {}
    for q in qs:
        item = {"type": q["type"], "instructions": q["instructions"]}
        if "criteria" in q:
            item["criteria"] = q["criteria"]
        out[q["id"]] = item
    return {"state": state, "questions": out}


def outcome_names(q: dict) -> list[str]:
    """The outcome names every adapter reports probabilities under."""
    if q["type"] == "noul":
        return ["yes", "no"]
    if q["type"] == "choice":
        return list(q["criteria"])
    return [str(i) for i in range(len(q["criteria"]))]


# ------------------------------------------------------------------------------------------------ context

FIRST = [
    "Maya",
    "Jonas",
    "Priya",
    "Tomasz",
    "Aiko",
    "Lucas",
    "Fatima",
    "Ben",
    "Chloe",
    "Diego",
    "Ingrid",
    "Kwame",
    "Lena",
    "Mateo",
    "Noor",
    "Oscar",
    "Rania",
    "Sven",
    "Tara",
    "Yusuf",
    "Elena",
    "Hugo",
    "Mei",
]
LAST = [
    "Okafor",
    "Lindqvist",
    "Raman",
    "Kowalski",
    "Tanaka",
    "Moreau",
    "Haddad",
    "Fischer",
    "Nguyen",
    "Alvarez",
    "Berg",
    "Mensah",
    "Novak",
    "Rossi",
    "Sato",
    "Walsh",
    "Kaur",
    "Duarte",
    "Ivanova",
]
AGENTS = [
    "Sam (Tier 1)",
    "Riley (Tier 1)",
    "Jordan (Tier 2)",
    "Alex (Billing)",
    "Casey (Logistics)",
    "Morgan (Platform)",
    "assistant-bot",
]
PRODUCTS = [
    "standing desk",
    "monitor arm",
    "ergonomic chair",
    "4K webcam",
    "USB-C dock",
    "noise-cancelling headset",
    "mechanical keyboard",
    "wireless mouse",
    "laptop stand",
    "cable tray kit",
    "desk lamp",
    "portable SSD",
]
PLANS = ["Starter", "Team", "Business", "Enterprise", "Pro Annual", "Pro Monthly"]
FEATURES = [
    "SSO login",
    "CSV export",
    "webhook delivery",
    "calendar sync",
    "mobile app",
    "invoice download",
    "API key rotation",
    "team invites",
    "two-factor setup",
    "usage dashboard",
    "Slack integration",
]
ERRORS = [
    "ERR_TIMEOUT",
    "HTTP 502",
    "HTTP 429",
    "SSL_HANDSHAKE_FAILED",
    "E_AUTH_EXPIRED",
    "QUOTA_EXCEEDED",
    "HTTP 500",
    "ECONNRESET",
    "VALIDATION_FAILED",
    "E_SYNC_CONFLICT",
]
CARRIERS = ["DHL", "UPS", "FedEx", "Royal Mail", "DPD", "PostNL", "USPS"]
CITIES = ["Manchester", "Leeds", "Rotterdam", "Lyon", "Austin", "Toronto", "Dublin", "Munich", "Porto", "Krakow"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October"]
SERVICES = ["billing-api", "sync-worker", "auth-gateway", "export-service", "notifier", "orders-db", "cdn-edge"]

CUSTOMER = [
    "I placed order {order} on {weekday} for a {product} and it still has not arrived.",
    "The tracking page from {carrier} has said 'in transit' for {n} days now.",
    "My card was charged {amount} twice on {date}, and I only made one purchase.",
    "Since the update on {weekday}, {feature} fails with {error} every time I try it.",
    "We are on the {plan} plan and {n} people on my team cannot log in this morning.",
    "The {product} arrived with a cracked corner and one of the screws missing.",
    "I already spoke to {agent} about this and was told it would be fixed within {n} business days.",
    "Honestly I am running out of patience, this is the third time I have had to write in.",
    "Could you confirm whether the invoice for {month} includes the seats we removed?",
    "I tried clearing the cache and reinstalling the app, and the problem is still there.",
    "Our finance team needs a corrected invoice before the end of the quarter.",
    "If this is not sorted by {weekday} we will have to look at other providers.",
    "To be fair, the replacement {product} you sent last time works perfectly.",
    "Is there a way to change the delivery address to our office in {city}?",
    "The export finished but the file only contains {n} rows instead of the full list.",
    "I would like my money back for the months the service did not work.",
    "Can someone call me? My number is on the account and I am free after {hour}:00.",
    "Thanks for the quick reply yesterday, that part is working now.",
    "We rely on {feature} for our weekly reporting, so this is blocking the whole team.",
    "The {carrier} driver left a note saying nobody was home, but I was at home all day.",
    "I noticed the price went up from {amount} to {amount2} without any notice.",
    "Please cancel the extra seat that was added by mistake on {date}.",
    "Attached is a screenshot of the error, it mentions {error} and request id {req}.",
    "I do not mind waiting a little, I just want to know what is happening.",
    "My manager has asked me to escalate this, so please treat it as a priority.",
    "The return label you emailed does not scan at the {carrier} drop-off point.",
]
AGENT = [
    "Thanks for reaching out, {name}. I have pulled up order {order} and I can see the delay.",
    "I am sorry about the double charge. I can see two authorisations of {amount} on {date}.",
    "I have escalated this to our {team} team with ticket reference {ticket}.",
    "Could you tell me which browser and app version you are using when you see {error}?",
    "I have issued a prepaid return label for the damaged {product}; it is attached to this email.",
    "Our logs show the {feature} job failed at {time} UTC; engineering is looking at it.",
    "As a gesture of goodwill I have applied a credit of {amount} to your next invoice.",
    "I checked with {carrier} and they expect delivery within {n} working days.",
    "The second charge is a pending authorisation and should drop off within {n} days.",
    "I have reset the SSO configuration for your workspace; please try logging in again.",
    "I understand how frustrating this is, and I will stay on this ticket until it is resolved.",
    "I am handing this over to {agent}, who handles {team} issues and will follow up today.",
]
NOTES = [
    "Internal note: customer is on {plan}, account age {n} months, lifetime value about {amount}.",
    "Internal note: duplicate of ticket {ticket}? Same order number, merged history below.",
    "Internal note: {service} had an incident between {time} and {time2} UTC, status page updated.",
    "Internal note: refund over {amount} needs a team lead approval, flagged to billing.",
    "Internal note: customer previously received a goodwill credit in {month}.",
    "Internal note: carrier claim opened with {carrier}, reference {req}.",
]
TOOLS = ["lookup_order", "get_payments", "search_logs", "get_account", "carrier_tracking", "list_invoices"]


def _fill(rng: random.Random, template: str, ctx: dict) -> str:
    amount = rng.randint(9, 480) + rng.choice([0, 0.5, 0.99, 0.49])
    values = {
        "order": ctx["order"],
        "product": ctx["product"],
        "carrier": ctx["carrier"],
        "plan": ctx["plan"],
        "name": ctx["first"],
        "weekday": rng.choice(WEEKDAYS),
        "month": rng.choice(MONTHS),
        "n": rng.randint(2, 14),
        "amount": f"${amount:.2f}",
        "amount2": f"${amount * 1.2:.2f}",
        "date": f"{rng.randint(1, 28)} {rng.choice(MONTHS)}",
        "feature": rng.choice(FEATURES),
        "error": rng.choice(ERRORS),
        "agent": rng.choice(AGENTS).split(" ")[0],
        "city": rng.choice(CITIES),
        "hour": rng.randint(9, 18),
        "req": f"req_{rng.getrandbits(40):010x}",
        "ticket": f"TCK-{rng.randint(10000, 99999)}",
        "team": rng.choice(["billing", "logistics", "platform"]),
        "time": f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}",
        "time2": f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}",
        "service": rng.choice(SERVICES),
    }
    return template.format(**values)


def _tool_call(rng: random.Random, ctx: dict) -> str:
    tool = rng.choice(TOOLS)
    if tool == "lookup_order":
        args = {"order_id": ctx["order"]}
        result = {
            "order_id": ctx["order"],
            "status": rng.choice(["shipped", "processing", "delayed", "delivered"]),
            "items": [{"sku": f"SKU-{rng.randint(1000, 9999)}", "name": ctx["product"], "qty": 1}],
            "total": round(rng.uniform(20, 900), 2),
            "placed_at": f"2026-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}",
        }
    elif tool == "get_payments":
        args = {"customer_id": ctx["customer_id"], "days": 30}
        result = {
            "payments": [
                {
                    "id": f"pay_{rng.getrandbits(32):08x}",
                    "amount": round(rng.uniform(9, 480), 2),
                    "state": rng.choice(["captured", "authorised", "refunded"]),
                }
                for _ in range(rng.randint(1, 3))
            ]
        }
    elif tool == "search_logs":
        args = {"service": rng.choice(SERVICES), "query": rng.choice(ERRORS), "window": "24h"}
        result = {
            "matches": rng.randint(0, 400),
            "first_seen": f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}Z",
            "sample": f"{rng.choice(SERVICES)} {rng.choice(ERRORS)} after {rng.randint(100, 30000)}ms",
        }
    elif tool == "get_account":
        args = {"customer_id": ctx["customer_id"]}
        result = {
            "plan": ctx["plan"],
            "seats": rng.randint(1, 80),
            "mrr": round(rng.uniform(10, 4000), 2),
            "open_tickets": rng.randint(0, 5),
            "nps": rng.randint(-100, 100),
        }
    elif tool == "carrier_tracking":
        args = {"carrier": ctx["carrier"], "tracking": f"{rng.getrandbits(48):012X}"}
        result = {
            "events": [
                {"at": f"{rng.choice(CITIES)} hub", "status": rng.choice(["arrived", "departed", "held", "exception"])}
                for _ in range(rng.randint(1, 4))
            ]
        }
    else:
        args = {"customer_id": ctx["customer_id"], "limit": 3}
        result = {
            "invoices": [
                {"month": rng.choice(MONTHS), "amount": round(rng.uniform(9, 900), 2), "paid": rng.random() > 0.3}
                for _ in range(rng.randint(1, 3))
            ]
        }
    return f"[tool_call] {tool}({json.dumps(args)})\n[tool_result] {json.dumps(result)}"


def _log_lines(rng: random.Random) -> str:
    lines = []
    for _ in range(rng.randint(2, 5)):
        level = rng.choice(["INFO", "WARN", "ERROR", "INFO", "DEBUG"])
        lines.append(
            f"2026-10-{rng.randint(1, 13):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:"
            f"{rng.randint(0, 59):02d}Z {level} {rng.choice(SERVICES)} "
            f"{rng.choice(ERRORS) if level in ('WARN', 'ERROR') else 'ok'} "
            f"latency_ms={rng.randint(3, 9000)} trace={rng.getrandbits(32):08x}"
        )
    return "\n".join(lines)


SHORT = 150  # below this many tokens a context is one customer message, as short requests are


def short_context(target_tokens: int, seed: int = 0) -> str:
    """One customer message of about target_tokens tokens: a short request, as decision models are
    often called (and as Liquid's one-question timings are)."""
    rng = random.Random(f"short-{seed}-{target_tokens}")
    ctx = {
        "first": rng.choice(FIRST),
        "last": rng.choice(LAST),
        "order": f"#{rng.randint(100000, 999999)}",
        "product": rng.choice(PRODUCTS),
        "carrier": rng.choice(CARRIERS),
        "plan": rng.choice(PLANS),
        "customer_id": f"cus_{rng.getrandbits(40):010x}",
    }
    target_chars, text = int(target_tokens * CHARS_PER_TOKEN), ""
    for template in rng.sample(CUSTOMER, len(CUSTOMER)):
        text = (text + " " + _fill(rng, template, ctx)).strip()
        if len(text) >= target_chars:
            break
    cut = text.rfind(" ", 0, target_chars)
    return text if len(text) <= target_chars else text[: cut if cut > 0 else target_chars]


def context(target_tokens: int, seed: int = 0) -> str:
    """A ticket thread of about target_tokens tokens, the same for every engine given (target, seed)."""
    if target_tokens < SHORT:
        return short_context(target_tokens, seed)
    rng = random.Random(f"ctx-{seed}-{target_tokens}")
    ctx = {
        "first": rng.choice(FIRST),
        "last": rng.choice(LAST),
        "order": f"#{rng.randint(100000, 999999)}",
        "product": rng.choice(PRODUCTS),
        "carrier": rng.choice(CARRIERS),
        "plan": rng.choice(PLANS),
        "customer_id": f"cus_{rng.getrandbits(40):010x}",
    }
    parts = [
        f"Ticket TCK-{rng.randint(10000, 99999)} | customer {ctx['first']} {ctx['last']} "
        f"({ctx['customer_id']}) | plan {ctx['plan']} | channel {rng.choice(['email', 'chat', 'phone transcript'])}"
    ]
    chars = len(parts[0])
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    turn = 0
    while chars < target_chars:
        turn += 1
        kind = rng.random()
        if kind < 0.38:
            body = " ".join(_fill(rng, t, ctx) for t in rng.sample(CUSTOMER, rng.randint(2, 5)))
            block = f"--- message {turn} from {ctx['first']} {ctx['last']} ---\n{body}"
        elif kind < 0.66:
            body = " ".join(_fill(rng, t, ctx) for t in rng.sample(AGENT, rng.randint(1, 3)))
            block = f"--- reply {turn} from {rng.choice(AGENTS)} ---\n{body}"
        elif kind < 0.84:
            block = _tool_call(rng, ctx)
        elif kind < 0.93:
            block = _log_lines(rng)
        else:
            block = _fill(rng, rng.choice(NOTES), ctx)
        parts.append(block)
        chars += len(block) + 2
    text = "\n\n".join(parts)
    cut = text.rfind(" ", 0, target_chars)
    return text[: cut if cut > 0 else target_chars]


def unique_id(seed: int, *key) -> str:
    """A short reference ID, deterministic per (seed, key), different for every request."""
    return f"{random.Random('id-' + '-'.join(map(str, (seed, *key)))).getrandbits(48):012x}"


def state(target_tokens: int, seed: int, *key) -> str:
    """The context with a unique reference at its start (defeats prefix caches across requests)."""
    return f"[ref {unique_id(seed, target_tokens, *key)}]\n{context(target_tokens, seed)}"


def plan(sizes, counts, n: int, warmup: int, seed: int = 0, scenario: str = "fresh") -> list[dict]:
    """Every request of a run, in order, the same list for every engine. fresh: `warmup` untimed requests of every
    (size, questions) shape (so first-shape kernel compilation never lands in a timed request), then n timed
    requests per shape. repeat and grow: per shape, one state with its own ID sent warmup + n times in a row, its
    `first` question one further on each time (repeat), or a message of about GROW tokens longer each time (grow);
    the first is the only one no prefix cache has seen."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {', '.join(SCENARIOS)}")
    out = []
    if scenario == "lines":
        for size in sizes:
            for nq in counts:
                for step in range(warmup + n):
                    phase, rep = ("warmup", step) if step < warmup else ("timed", step - warmup)
                    text = numbered(state(size, seed, phase, nq, rep), nq)
                    out.append({"phase": phase, "size": size, "nq": nq, "rep": rep, "state": text, "lines": True})
        return out
    if scenario != "fresh":
        for size in sizes:
            for nq in counts:
                text = state(size, seed, scenario, nq)
                for step in range(warmup + n):
                    if scenario == "grow":
                        text += "\n" + short_context(GROW, seed * 1_000_003 + step)
                    phase, rep = ("warmup", step) if step < warmup else ("timed", step - warmup)
                    first = step if scenario == "repeat" else 0
                    out.append({"phase": phase, "size": size, "nq": nq, "rep": rep, "state": text, "first": first})
        return out
    for size in sizes:
        for nq in counts:
            for w in range(warmup):
                out.append(
                    {"phase": "warmup", "size": size, "nq": nq, "rep": w, "state": state(size, seed, "warmup", nq, w)}
                )
    for size in sizes:
        for nq in counts:
            for rep in range(n):
                out.append(
                    {"phase": "timed", "size": size, "nq": nq, "rep": rep, "state": state(size, seed, "timed", nq, rep)}
                )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--show", type=int, help="print the context of this size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dump", help="write every request of a default run (n=5, warmup=2) as JSON")
    args = p.parse_args()
    if args.show:
        text = state(args.show, args.seed, "show")
        print(text)
        print(f"\n# {len(text.split())} words, {len(text)} chars")
    if args.dump:
        reqs = plan(SIZES, QUESTION_COUNTS, 5, 2, args.seed)
        for r in reqs:
            qs = questions(r["nq"], r.get("first", 0))
            r["systemone"] = to_systemone(r["state"], qs)
        with open(args.dump, "w") as f:
            json.dump(reqs, f)
        print(f"wrote {len(reqs)} requests to {args.dump}")


if __name__ == "__main__":
    main()
