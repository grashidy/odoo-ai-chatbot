import xmlrpc.client, json, os, time, re, logging, threading, io, base64, socket
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from openai import OpenAI
try:
    from groq import Groq as _GroqClient  # kept only for Whisper voice transcription
except ImportError:
    _GroqClient = None
try:
    import requests as _requests
except ImportError:
    _requests = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Global socket timeout prevents xmlrpc calls from hanging indefinitely
socket.setdefaulttimeout(55)

# ── AI provider: Gemini (primary) with Groq fallback for Whisper ───────────────
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
AI_MODEL        = "gemini-1.5-flash"

# Groq key — only used for Whisper audio transcription
_key_file = Path(__file__).parent / ".groq_key"
DEFAULT_GROQ_KEY = (
    os.environ.get("GROQ_API_KEY", "").strip()
    or (_key_file.read_text(encoding="utf-8").strip() if _key_file.exists() else "")
)

def _make_chat_client(override_key=None):
    """Groq first (reliable tool calling), Gemini as fallback."""
    if DEFAULT_GROQ_KEY:
        return OpenAI(api_key=DEFAULT_GROQ_KEY, base_url="https://api.groq.com/openai/v1"), "llama-3.1-8b-instant"
    if GEMINI_API_KEY:
        return OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL), "gemini-2.0-flash"
    if override_key:
        return OpenAI(api_key=override_key, base_url="https://api.groq.com/openai/v1"), "llama-3.1-8b-instant"
    return None, None

# ── Odoo connection ────────────────────────────────────────────────────────────
# Set ODOO_API_KEY as a Railway environment variable (Variables tab) — never hardcode it.
ODOO_URL     = os.environ.get("ODOO_URL",     "https://alforat-beta.odoo.com").rstrip("/")
ODOO_DB      = os.environ.get("ODOO_DB",      "alforat-beta-35112787")
ODOO_UID     = int(os.environ.get("ODOO_UID", "2"))
ODOO_API_KEY = os.environ.get("ODOO_API_KEY", "")

logging.info("Odoo: %s  db=%s  uid=%d", ODOO_URL, ODOO_DB, ODOO_UID)

_odoo = xmlrpc.client.ServerProxy(ODOO_URL + "/xmlrpc/2/object")

# Multi-company context for construction company (company_id=3)
ODOO_CTX = {"allowed_company_ids": [1, 2, 3, 4, 5], "company_id": 3}

def odoo_call(model, method, args, kwargs=None):
    kw = dict(kwargs) if kwargs else {}
    if "context" not in kw:
        kw["context"] = ODOO_CTX
    return _odoo.execute_kw(ODOO_DB, ODOO_UID, ODOO_API_KEY,
                             model, method, args, kw)

# ── Tool definitions (OpenAI/Groq format) ─────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "odoo_search",
            "description": "Get records from Odoo (employees, units, contracts, projects, partners…).",
            "parameters": {
                "type": "object",
                "properties": {
                    "model":  {"type": "string", "description": "Odoo model. E.g. project.subcontracting.boq.line, boq.contract, construction.advance.payment, project.detailed.item.line, purchase.order, hr.employee"},
                    "domain": {"anyOf": [{"type": "string"}, {"type": "array"}], "description": "Filter domain. Use [] for all records. E.g. [[\"state\",\"=\",\"sale\"]]"},
                    "fields": {"type": "array", "items": {"type": "string"}, "description": "List of field names to return. E.g. [\"name\",\"state\",\"project_id\"]"},
                    "limit":  {"type": "integer", "description": "Max rows (default 50, max 200)"},
                    "order":  {"type": "string",  "description": "Sort. E.g. 'boq_cost desc'"}
                },
                "required": ["model", "domain", "fields"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "odoo_count",
            "description": "Count records in an Odoo model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model":  {"type": "string"},
                    "domain": {"anyOf": [{"type": "string"}, {"type": "array"}], "description": "Filter domain. Use [] for all records."}
                },
                "required": ["model", "domain"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "odoo_get_fields",
            "description": "List field names/types for an Odoo model. Use when unsure of field names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"}
                },
                "required": ["model"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "odoo_read_group",
            "description": "Group & aggregate Odoo records. Use for 'how many X per Y' or totals. Count is returned automatically as 'count' — never include 'id:count' in aggregates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Odoo model name."},
                    "domain": {"anyOf": [{"type": "string"}, {"type": "array"}], "description": "Filter domain. Use [] for all records. E.g. [[\"state\",\"=\",\"draft\"]]"},
                    "groupby": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "string"}
                        ],
                        "description": "Fields to group by. E.g. [\"project_id\"] or [\"state\"] or [\"partner_id\"]."
                    },
                    "aggregates": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "string"}
                        ],
                        "description": "Numeric fields to sum. E.g. [\"boq_cost:sum\",\"quantity:sum\"]. Use [] if only need count. NEVER include 'id:count' — count is automatic."
                    }
                },
                "required": ["model", "domain", "groupby"]
            }
        }
    }
]

# ── Tool execution ─────────────────────────────────────────────────────────────
def _coerce_domain(raw):
    """Ensure domain is always a list, even when the LLM passes a string."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if raw in ("", "[]", "None", "null"):
            return []
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []

def _coerce_list(raw):
    """Ensure value is a list of strings, even when LLM passes a JSON string."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if raw in ("", "[]", "None", "null"):
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, str):
                return [parsed]
        except Exception:
            pass
        # bare single field name like "project_id"
        return [raw] if raw else []
    return []

_MAX_RESULT_CHARS  = 12000  # max total chars returned to AI per tool call
_MAX_FIELD_CHARS   = 180    # max chars for any single string field value (long Arabic names)

def _trim_long_fields(rec):
    """Trim individual string fields that are very long (e.g. long Arabic item descriptions)."""
    if not isinstance(rec, dict):
        return rec
    out = {}
    for k, v in rec.items():
        if isinstance(v, str) and len(v) > _MAX_FIELD_CHARS:
            out[k] = v[:_MAX_FIELD_CHARS] + '…'
        else:
            out[k] = v
    return out

def _truncate_result(text):
    if len(text) > _MAX_RESULT_CHARS:
        return text[:_MAX_RESULT_CHARS] + f'\n... [truncated — {len(text)} chars total, showing first {_MAX_RESULT_CHARS}]'
    return text

def _flatten_m2o(val):
    """Convert Odoo many2one [id, "Name"] tuples to just the display name string."""
    if isinstance(val, (list, tuple)) and len(val) == 2 and isinstance(val[0], int) and isinstance(val[1], str):
        return val[1]
    return val

def _flatten_record(rec):
    """Flatten all many2one fields in a record dict so the AI sees plain strings."""
    if isinstance(rec, dict):
        return {k: _flatten_m2o(v) for k, v in rec.items()}
    return rec

def run_tool(name, args):
    try:
        if name == "odoo_search":
            domain = _coerce_domain(args.get("domain", []))
            limit  = min(int(args.get("limit", 30)), 100)   # reduced: 30 default, 100 max
            raw_fields = args.get("fields", ["display_name"])
            fields = raw_fields if isinstance(raw_fields, list) else _coerce_list(raw_fields)
            if not fields:
                fields = ["display_name"]
            kwargs = {"fields": fields, "limit": limit}
            if args.get("order"):
                kwargs["order"] = args["order"]
            results = odoo_call(args["model"], "search_read", [domain], kwargs)
            flat = [_trim_long_fields(_flatten_record(r)) for r in results]
            return _truncate_result(json.dumps(flat, ensure_ascii=False, default=str))

        elif name == "odoo_count":
            domain = _coerce_domain(args.get("domain", []))
            count  = odoo_call(args["model"], "search_count", [domain])
            return json.dumps({"count": count})

        elif name == "odoo_get_fields":
            fields = odoo_call(args["model"], "fields_get", [],
                               {"attributes": ["string", "type"]})
            simple = {
                k: {"type": v["type"], "label": v["string"]}
                for k, v in fields.items()
                if v["type"] not in ("binary", "html", "serialized")
                and not k.startswith("message_")
                and not k.startswith("activity_")
            }
            return json.dumps(simple, ensure_ascii=False)

        elif name == "odoo_read_group":
            domain     = _coerce_domain(args.get("domain", []))
            groupby    = _coerce_list(args.get("groupby", []))
            agg_fields = _coerce_list(args.get("aggregates", []))
            # Strip invalid aggregates — Odoo count is automatic (__count), never pass id:count or bare 'count'
            agg_fields = [f for f in agg_fields if f not in ("id:count", "id", "__count", "count")]
            fields_list = list(groupby) + list(agg_fields)
            result = odoo_call(
                args["model"], "read_group",
                [domain, fields_list, groupby],
                {"lazy": False}
            )
            cleaned = []
            for row in result:
                item = {}
                for k, v in row.items():
                    if k == "__count":
                        item["count"] = v
                    elif k == "__domain":
                        continue
                    else:
                        item[k] = _flatten_m2o(v)
                cleaned.append(item)
            return _truncate_result(json.dumps(cleaned, ensure_ascii=False, default=str))

    except Exception as e:
        err = str(e)
        if "Access Denied" in err or "Fault 3" in err:
            return json.dumps({
                "error": "Odoo Access Denied — the API key is expired or invalid.",
                "fix": "Go to Railway → Variables and update ODOO_API_KEY with a valid Odoo API key (Settings → Technical → API Keys in Odoo)."
            })
        return json.dumps({"error": err, "hint": "Try odoo_get_fields to check available fields"})

# ── System prompt ──────────────────────────────────────────────────────────────
def get_system_prompt():
    today_date = _date.today().isoformat()   # e.g. "2026-08-05"
    return f"""You are an AI assistant for Odoo 18 (Real Estate, Construction, HR, Procurement). Reply in the same language as the user (Arabic or English).

TODAY = {today_date}  ← use this exact string for any date comparison domains (e.g. [['date_to','<','{today_date}']])

RULES:
- Use tools for real data. Never guess or fabricate.
- BATCH: Call ALL needed tools in ONE response — never chain across turns.
- After tool results arrive, answer immediately. Do NOT call more tools.
- NEVER call the same model more than once per turn — pick the right fields the first time.
- Totals/counts → odoo_read_group. Lists → odoo_search (limit 20).
- Format numbers with commas. Currency = EGP. Use markdown tables.
- On field error → odoo_get_fields once, then retry. Multi-company context is automatic.

CHARTS: CHART_BAR:{{"title":"T","labels":["A","B"],"data":[10,20]}}  CHART_PIE:{{"title":"T","labels":["A"],"data":[1]}}

══ CONSTRUCTION MODELS — use ONLY these for any construction/project question ══

project.project  →  construction projects
  Fields: name, user_id, partner_id, date_start, date (planned end),
          expected_end_date, actual_end_date, state, last_update_status,
          total_planned_cost, total_actual_cost
  NOTE: fetch ALL these fields in ONE call — never call project.project twice

main.project.plan.operation  →  construction plan milestones (top-level grouping)
  Fields: name, project_id, plan_type, user_id, sub_plan_count
  plan_type values: subcontract | raw_material | assets | labor | misc
  ⚠ WARNING: the "date" field = system creation timestamp (auto-set = same as create_date) — NOT a user-set deadline
  ⚠ NEVER use "date" on this model to determine behind-schedule status — it is meaningless for scheduling
  NOTE: NO state/progress/completion field — cannot determine done/overdue from this model alone
  → For "behind schedule" milestones: query sub.project.plan.operation and use main_plan_id to identify parent

sub.project.plan.operation  →  sub-operations under a milestone — ONLY model with real date ranges
  Fields: name, project_id, main_plan_id, date_from (Period From), date_to (Period To), plan_type, user_id
  date_from / date_to = user-set planned work period dates (YYYY-MM-DD format, set by project managers)
  ▶ "Behind schedule / overdue" = date_to < TODAY ({today_date}) AND date_to != False
  Domain for overdue: [{{"date_to","<","{today_date}"}},{{"date_to","!=",False}}]
  NOTE: NO state/progress/completion field on this model either

project.detailed.item.line  →  client BOQ line items
  Fields: name, project_id, quantity, unit_cost, total_cost (=Total Price), main_item_line_id, boq_item_id

project.subcontracting.boq.line  →  subcontractor BOQ items
  Fields: name, project_id, quantity, unit_cost, billed_qty, remain_qty, boq_cost
  WARNING: boq_cost = "BOQ Unit Price" — always 0, do NOT use for financial totals

project.main.item.line  →  BOQ section/category headers
  Fields: name, project_id

subcontractor.contract  →  subcontractor contracts — PRIMARY financial model
  Fields: name, project_id, partner_id, contract_value (=THE contract total),
          bills_amount_total, bills_amount_due, total_adv_amount
  status field values: draft | running | closed   (field name is "status" NOT "state")

subcontractor.contract.line  →  individual contract line items
  Fields: name, contract_id, project_id, unit_price, total_price, billed_qty, remain_qty

boq.contract  →  BOQ scope config record — EMPTY in database (0 records), NO financial data
  Fields: name, project_id, partner_id
  ⚠ NEVER use boq.contract for any financial or contract-value queries — it has no money fields and no data

construction.advance.payment  →  advance payments to contractors/clients
  Fields: name, partner_id, amount, state, project_id, due_amount, settled_amount, payment_type
  state values: draft | run | settled   (NOT "running")
  payment_type values: customer | subcontractor

project.task  →  project tasks
  Fields: name, project_id, stage_id, date_deadline, kanban_state

project.boq.item  →  global BOQ item catalogue
  Fields: name, code, uom_id, billing_type (billable|unbillable)

══ REAL ESTATE MODELS — use ONLY for RE/property questions, NEVER for construction ══

rs.dev.unit  →  RE property units for sale (USE THIS, not project.rs.unit)
  Fields: unit_code, rs_project_id, rs_building_id, phase_id, unit_price,
          sold_price, net_area, paid_amount, price_after_discount,
          usability_type (commercial|administrative|residential|medical|parking_slot),
          unit_type (apartment|duplex|villa|twin_house|town_house),
          historical_process (holded|booked|reserved|contracted)

project.rs.building  →  RE buildings (name, project_id)
project.rs.phase     →  RE development phases (name, project_id) ← NOT for construction

══ SHARED MODELS ══
purchase.order      → purchase orders
  Fields: name, partner_id, state, amount_total, date_order
  state values: draft=RFQ | sent=RFQ Sent | to approve=To Approve | purchase=Purchase Order | done=Locked | cancel=Cancelled

purchase.order.line → PO lines (order_id,product_id,product_qty,price_unit,price_subtotal)
hr.employee         → employees (name,department_id,job_title,job_id,active,work_phone,work_email)
hr.department       → departments (name,manager_id)
res.partner         → partners/vendors/clients (name,phone,email,is_company)

══ ROUTING — which model to answer each question ══
• projects / list of projects / مشاريع                       → project.project (NEVER project.rs.phase)
• plan / milestones / schedule / خطة / مراحل                 → main.project.plan.operation + sub.project.plan.operation
• behind schedule / overdue / متأخر / delay / تأخير          → sub.project.plan.operation domain [{{"date_to","<","{today_date}"}},{{"date_to","!=",False}}]
                                                               then identify parent milestone via main_plan_id field
• client BOQ / مقايسة العميل                                 → project.detailed.item.line (total_cost for amounts)
• subcontractor BOQ items / بنود الباطن                      → project.subcontracting.boq.line
• BOQ contracts / subcontractor contracts / عقود المقاولين   → subcontractor.contract (use contract_value)
• contract value / contract total / قيمة العقد               → subcontractor.contract (use contract_value)
• advance payments / مدفوعات مقدمة                          → construction.advance.payment
• RE units / وحدات عقارية / شقق / عقارات                    → rs.dev.unit
• purchases / مشتريات                                       → purchase.order
• employees / موظفين                                        → hr.employee

CRITICAL RULES:
- project.rs.phase and project.rs.unit are REAL ESTATE ONLY — NEVER for construction
- rs.dev.unit is the correct RE sales model (project.rs.unit has NO price/status data)
- For ALL contract/financial queries: use subcontractor.contract with contract_value field
- boq.contract is EMPTY (no records) — NEVER use it for any query, it will return nothing
- construction.advance.payment state = "run" (not "running")
- COUNTING: odoo_read_group aggregates=[] gives count automatically; never add id:count
- Domain cannot compare two fields — fetch rows and filter in the answer text
- Call each model ONLY ONCE per turn — include all needed fields in a single call
- Many2one fields return as plain strings (e.g. project_id = "INFINITY MALL - Phase 1") — use them directly in the table
- BEHIND SCHEDULE: ONLY sub.project.plan.operation has real date ranges; use domain [{{"date_to","<","{today_date}"}},{{"date_to","!=",False}}]
  main.project.plan.operation.date = creation timestamp only — NEVER use it for overdue/schedule checks"""

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Support running under /ai path on the server
app.config["APPLICATION_ROOT"] = "/"

@app.route("/")
@app.route("/ai")
@app.route("/ai/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
@app.route("/ai/chat", methods=["POST"])
def chat():
    data = request.json
    history = data.get("messages", [])
    override_key = data.get("api_key", "").strip() or None
    client, _ai_model = _make_chat_client(override_key)

    if not client:
        return jsonify({"error": "AI API key is required. Set GEMINI_API_KEY in Railway Variables."}), 400

    def generate():
        try:
            # Keep only last 4 messages from history to limit token usage
            recent = history[-4:] if len(history) > 4 else history
            messages = [{"role": "system", "content": get_system_prompt()}]
            for m in recent:
                messages.append({"role": m["role"], "content": m.get("content") or ""})

            max_iterations   = 5
            tool_call_counts = {}   # tool_name → how many times called this turn
            tool_fail_count  = 0   # consecutive schema-validation failures
            answered         = False

            for iteration in range(max_iterations):
                response = None
                # Run Groq call in a thread so we can send SSE keepalive pings
                # while waiting — Railway drops idle connections after ~60 s
                _result  = [None]
                _api_err = [None]
                _done    = threading.Event()

                # Gemini requires tool definitions present whenever the history
                # contains tool results. Use tool_choice="none" to block new calls
                # instead of omitting tools entirely.
                _has_tool_results = any(m.get("role") == "tool" for m in messages)
                def _groq_call():
                    try:
                        call_kw = dict(
                            model=_ai_model,
                            messages=messages,
                            tools=TOOLS,
                            tool_choice="none" if _has_tool_results else "auto",
                            max_tokens=1800,
                            temperature=0.1,
                        )
                        _result[0] = client.chat.completions.create(**call_kw)
                    except Exception as e:
                        _api_err[0] = e
                    finally:
                        _done.set()

                threading.Thread(target=_groq_call, daemon=True).start()
                # Ping every 20 s so Railway doesn't kill the idle connection
                while not _done.wait(timeout=20):
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"

                response = _result[0]
                api_err  = _api_err[0]
                try:
                    if api_err is not None:
                        raise api_err
                except Exception as api_err:
                    err_msg = str(api_err)
                    is_rate      = "rate_limit" in err_msg.lower() or "429" in err_msg
                    is_tool_fail = "tool_use_failed" in err_msg or "tool call validation" in err_msg.lower()
                    if is_rate:
                        # Parse Groq wait time — handles "5s", "1m5s", "1m5.000s"
                        _m_min = re.search(r'try again in (\d+)m([\d.]+)s', err_msg, re.IGNORECASE)
                        _m_sec = re.search(r'try again in ([\d.]+)s', err_msg, re.IGNORECASE)
                        if _m_min:
                            wait_sec = int(_m_min.group(1)) * 60 + int(float(_m_min.group(2))) + 2
                        elif _m_sec:
                            wait_sec = int(float(_m_sec.group(1))) + 2
                        else:
                            wait_sec = 65
                        if "per day" in err_msg.lower() or "tpd" in err_msg.lower() or wait_sec > 300:
                            yield f"data: {json.dumps({'type': 'text', 'text': '⚠️ Daily API quota exhausted. Please wait a few hours or use a new Groq API key (free at console.groq.com).'})}\n\n"
                            answered = True
                            break
                        else:
                            yield f"data: {json.dumps({'type': 'tool', 'name': 'wait', 'input': {'model': f'Waiting {wait_sec}s for rate limit to reset...'}})}\n\n"
                            # Yield pings during wait so Railway doesn't drop the connection
                            elapsed = 0
                            while elapsed < wait_sec:
                                chunk = min(20, wait_sec - elapsed)
                                time.sleep(chunk)
                                elapsed += chunk
                                if elapsed < wait_sec:
                                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                            continue  # retry this iteration
                    elif is_tool_fail:
                        tool_fail_count += 1
                        logging.warning("tool_use_failed #%d: %s", tool_fail_count, err_msg[:400])
                        if tool_fail_count >= 2:
                            yield f"data: {json.dumps({'type': 'text', 'text': '⚠️ The AI model repeatedly sent invalid tool parameters. Please rephrase your question or try again.'})}\n\n"
                            answered = True
                            break
                        # Inject correction hint with correct type guidance and retry
                        schema_hint = (
                            "Your last tool call was rejected because a parameter had the wrong type. "
                            "Fix the types and retry: "
                            "fields must be a JSON array like [\"name\",\"state\"]; "
                            "domain must be a JSON array like [] or [[\"state\",\"=\",\"draft\"]]; "
                            "groupby must be a JSON array like [\"project_id\"]; "
                            "aggregates must be a JSON array like [\"boq_cost:sum\"] or []."
                        )
                        messages.append({"role": "user", "content": schema_hint})
                        continue
                    else:
                        yield f"data: {json.dumps({'type': 'text', 'text': f'❌ API Error: {err_msg[:250]}'})}\n\n"
                        answered = True
                    break

                if response is None:
                    yield f"data: {json.dumps({'type': 'text', 'text': '⚠️ No response from AI. Please try again.'})}\n\n"
                    answered = True
                    break

                tool_fail_count = 0  # reset on successful API call

                msg    = response.choices[0].message
                finish = response.choices[0].finish_reason

                # Build assistant message — NEVER include tool_calls key if empty
                asst_msg = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    asst_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                messages.append(asst_msg)

                # No tool calls → final text answer
                if finish in ("stop", "end_turn") or not msg.tool_calls:
                    text = msg.content or "I was unable to generate a response. Please try again."
                    # Guard: if model leaked raw function-call syntax, suppress it
                    if "<function=" in text or "odoo_search" in text[:50]:
                        logging.warning("Model leaked raw tool call in content, retrying synthesis")
                        # Inject stronger instruction and continue to next iteration
                        messages.append({"role": "user", "content": "IMPORTANT: Do NOT output function call syntax. Write only the final human-readable answer in the user's language."})
                        answered = False
                        continue
                    yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"
                    answered = True
                    break

                # ── Parse args + deduplicate within this batch ─────────────────
                # If the AI requests the same (tool, model) more than once in a
                # single response, only execute the FIRST — skip the rest.
                _seen_keys = set()
                _batch = []   # (tc, args, call_key, is_dup)
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {}
                    _model_arg = args.get("model", "") if isinstance(args, dict) else ""
                    _call_key  = f"{tc.function.name}:{_model_arg}"
                    is_dup = _call_key in _seen_keys
                    _seen_keys.add(_call_key)
                    _batch.append((tc, args, _call_key, is_dup))

                # Notify client and count only real (non-dup) calls
                for tc, args, _call_key, is_dup in _batch:
                    if not is_dup:
                        yield f"data: {json.dumps({'type': 'tool', 'name': tc.function.name, 'input': args})}\n\n"
                        tool_call_counts[_call_key] = tool_call_counts.get(_call_key, 0) + 1
                    else:
                        logging.info("Deduped duplicate tool call: %s", _call_key)

                # Cross-iteration loop = same (tool, model) called in two separate turns
                loop_detected = any(v >= 2 for v in tool_call_counts.values())

                # ── Execute all non-dup tools in PARALLEL ──────────────────────
                _tool_results = {}   # tc.id → result JSON string
                _t_lock   = threading.Lock()
                _pending  = [sum(1 for _, _, _, d in _batch if not d)]
                _all_done = threading.Event()
                if _pending[0] == 0:
                    _all_done.set()

                def _run_one(n, a, tid):
                    try:
                        res = run_tool(n, a)
                    except Exception as _te:
                        res = json.dumps({"error": str(_te)})
                    with _t_lock:
                        _tool_results[tid] = res
                        _pending[0] -= 1
                        if _pending[0] == 0:
                            _all_done.set()

                for tc, args, _call_key, is_dup in _batch:
                    if not is_dup:
                        threading.Thread(target=_run_one,
                                         args=(tc.function.name, args, tc.id),
                                         daemon=True).start()

                # Wait for all parallel calls (up to 45 s), SSE ping every 5 s
                _waited = 0
                while _waited < 45 and not _all_done.is_set():
                    _all_done.wait(timeout=5)
                    _waited += 5
                    if not _all_done.is_set() and _waited < 45:
                        yield f"data: {json.dumps({'type': 'ping'})}\n\n"

                # Build one tool-result message per tool_call_id the assistant sent
                for tc, args, _call_key, is_dup in _batch:
                    if is_dup:
                        content = json.dumps({"note": "duplicate query skipped — reuse the result already returned for this model"})
                    else:
                        content = _tool_results.get(tc.id)
                        if not content:
                            content = json.dumps({"error": "Odoo query timed out after 45s. Try a simpler query."})
                            logging.warning("Tool %s timed out after 45 s", tc.function.name)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

                # Inject "answer now" so the next iteration produces text, not more tool calls
                messages.append({
                    "role": "user",
                    "content": "The Odoo data above has been retrieved. Give the final answer now in the same language as the original question. Use markdown tables. Do NOT output any function call syntax or tool invocations — just the answer."
                })

                if loop_detected:
                    _fr = [None, None]; _fe = threading.Event()
                    def _final_call():
                        try:
                            _fr[0] = client.chat.completions.create(
                                model=_ai_model,
                                messages=messages,
                                max_tokens=1800,
                                temperature=0.1,
                            )
                        except Exception as _fce:
                            _fr[1] = str(_fce)
                        finally:
                            _fe.set()
                    threading.Thread(target=_final_call, daemon=True).start()
                    while not _fe.wait(timeout=30):
                        yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                    if _fr[0]:
                        text = _fr[0].choices[0].message.content or "⚠️ Empty response. Please try again."
                    elif _fr[1]:
                        text = f"⚠️ AI Error: {_fr[1][:300]}"
                    else:
                        text = "⚠️ No response received. Please try again."
                    yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"
                    answered = True
                    break

            # Fallback: loop exhausted all iterations without producing an answer
            if not answered:
                yield f"data: {json.dumps({'type': 'text', 'text': '⚠️ Could not complete the request after several attempts. Please rephrase and try again.'})}\n\n"

        except Exception as outer_err:
            # Last-resort catch — always send something to unblock the UI
            yield f"data: {json.dumps({'type': 'text', 'text': f'❌ Unexpected error: {str(outer_err)[:300]}'})}\n\n"

        finally:
            # Always send done so the frontend never hangs
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/reports-data")
@app.route("/ai/reports-data")
def reports_data():
    def safe(model, method, args, kwargs=None):
        try:
            return odoo_call(model, method, args, kwargs or {})
        except Exception as e:
            return {"error": str(e)}

    # Construction: BOQ lines by project
    boq_by_project = safe("project.subcontracting.boq.line", "read_group",
        [[], ["project_id", "boq_cost:sum", "quantity:sum"], ["project_id"]], {"lazy": False})
    # Construction: subcontractor contracts by project
    boq_contracts_by_project = safe("subcontractor.contract", "read_group",
        [[], ["project_id", "contract_value:sum", "bills_amount_total:sum"], ["project_id"]], {"lazy": False})
    # Construction: advance payments by state
    adv_by_state = safe("construction.advance.payment", "read_group",
        [[], ["state"], ["state"]], {"lazy": False})
    # Construction: advance payments by project
    adv_by_project = safe("construction.advance.payment", "read_group",
        [[], ["project_id", "amount:sum", "due_amount:sum"], ["project_id"]], {"lazy": False})
    # Construction: detailed items progress by project
    items_by_project = safe("project.detailed.item.line", "read_group",
        [[], ["project_id", "total_cost:sum", "actual_cost:sum"], ["project_id"]], {"lazy": False})
    # Purchases: top 10 vendors by total
    po_by_vendor = safe("purchase.order", "read_group",
        [[["state", "in", ["purchase", "done"]]], ["partner_id", "amount_total:sum"], ["partner_id"]],
        {"lazy": False, "limit": 10, "orderby": "amount_total desc"})
    # HR: employees by department
    emp_by_dept = safe("hr.employee", "read_group",
        [[], ["department_id"], ["department_id"]], {"lazy": False})
    # Real Estate: units by usability_type
    units_by_type = safe("rs.dev.unit", "read_group",
        [[], ["usability_type"], ["usability_type"]], {"lazy": False})
    # Construction: detailed BOQ lines by project (client-facing BOQ) — total_cost is the correct amount
    boq_detail_by_project = safe("project.detailed.item.line", "read_group",
        [[], ["project_id", "total_cost:sum"], ["project_id"]], {"lazy": False})
    # Construction: plan milestones by plan_type
    plan_by_type = safe("main.project.plan.operation", "read_group",
        [[], ["plan_type"], ["plan_type"]], {"lazy": False})
    # Real Estate: units by historical_process (correct status field on rs.dev.unit)
    units_by_hp = safe("rs.dev.unit", "read_group",
        [[], ["historical_process"], ["historical_process"]], {"lazy": False})

    def extract(rows, label_field, count_field="__count", amount_field=None):
        if isinstance(rows, dict) and "error" in rows:
            return {"error": rows["error"]}
        out = []
        for r in rows:
            lbl = r.get(label_field)
            if isinstance(lbl, (list, tuple)):
                lbl = lbl[1]
            elif not lbl:
                lbl = "غير محدد"
            entry = {"label": str(lbl), "count": r.get(count_field, 0)}
            if amount_field:
                entry["amount"] = r.get(amount_field, 0)
            out.append(entry)
        return out

    return jsonify({
        "boq_by_project":           extract(boq_by_project,           "project_id",        amount_field="boq_cost"),
        "boq_contracts_by_project": extract(boq_contracts_by_project, "project_id",        amount_field="contract_value"),
        "boq_detail_by_project":    extract(boq_detail_by_project,    "project_id",        amount_field="total_cost"),
        "adv_by_state":             extract(adv_by_state,             "state"),
        "adv_by_project":           extract(adv_by_project,           "project_id",        amount_field="amount"),
        "items_by_project":         extract(items_by_project,         "project_id",        amount_field="total_cost"),
        "po_by_vendor":             extract(po_by_vendor,             "partner_id",        amount_field="amount_total"),
        "emp_by_dept":              extract(emp_by_dept,              "department_id"),
        "units_by_state":           extract(units_by_hp,              "historical_process"),
        "units_by_type":            extract(units_by_type,            "usability_type"),
        "plan_by_state":            extract(plan_by_type,             "plan_type"),
    })

# ── BOQ Import ─────────────────────────────────────────────────────────────────

@app.route("/upload-boq")
@app.route("/ai/upload-boq")
def upload_boq_page():
    return render_template("upload_boq.html")

@app.route("/parse-boq", methods=["POST"])
@app.route("/ai/parse-boq", methods=["POST"])
def parse_boq():
    """Read uploaded Excel, extract rows, use AI to suggest column mapping."""
    try:
        return _parse_boq_impl()
    except Exception as e:
        logging.exception("Unhandled error in parse_boq")
        return jsonify({"error": f"Server error: {e}"}), 500

import unicodedata as _ucd

def _norm_header(h):
    """Normalize Arabic header: alef variants, alef-maqsura to ya, ta-marbuta, harakat, invisible chars."""
    h = _ucd.normalize("NFKC", str(h))
    cleaned = []
    for ch in h:
        cp = ord(ch)
        if (0x200B <= cp <= 0x200F or 0x202A <= cp <= 0x202E or
                0x2060 <= cp <= 0x2069 or cp in (0xFEFF, 0xA0)):
            cleaned.append(" ")
        else:
            cleaned.append(ch)
    h = "".join(cleaned)
    h = re.sub("[\u0623\u0625\u0622\u0671]", "\u0627", h)  # alef variants -> bare alef
    h = h.replace("\u0649", "\u064A")  # alef maqsura (\u0649) -> ya (\u064A)
    h = h.replace("\u0629", "\u0647")  # ta marbuta -> ha
    h = re.sub("[\u064B-\u065F]", "", h)  # strip harakat
    return h.strip().lower()

_BOQ_KEYWORD_RULES = [
    # item_code MUST come before name so "item code"/"item no" headers are claimed first,
    # preventing "item" (added to name keywords below) from matching them.
    ("item_code",         ["القسم", "قسم", "رقم البند",
                            "item code", "item no", "task code", "line code", "line no", "serial"]),
    # name comes second; "item" / "items" / "item description" are common English column names
    ("name",              ["البند", "بيان", "الوصف", "وصف",
                            "description", "task code description", "work description",
                            "item name", "item description", "activity",
                            "item", "items"]),
    ("quantity",          ["الكمية", "كمية", "كميه", "quantity", "qty"]),
    ("unit",              ["الوحدة", "وحدة", "وحده",
                            "description uom", "uom", "unit of measure", "task code unit", "unit"]),
    ("unit_cost",         ["الفئة", "فئة", "سعر الوحدة", "سعر",
                            "unit cost", "unit price", "unit rate", "rate", "boq rate"]),
    ("total_cost",        ["الاجمالي", "اجمالي", "المبلغ",
                            "total", "amount", "subtotal", "total cost", "total amount", "boq amount"]),
    ("main_item_line_id", ["main item line id"]),
    ("section_name",      ["chapter name", "section header", "chapter title", "main chapter"]),
    ("boq_item_id",       ["description id", "boq item id"]),
    ("work_type",         ["نوع العمل", "description categ", "categ", "work type", "category",
                            "trade", "division", "discipline"]),
    ("notes",             ["ملاحظات", "resource description", "notes", "remarks"]),
]
_BOQ_SKIP_KEYWORDS = ["#n/a", "cost code", "wastage", "usage", "resource code",
                       "billed", "remain", "col0"]

# Keywords that flag a sheet as a resource-analysis sheet (not a BOQ sheet)
_RESOURCE_SHEET_PENALTIES = ["resource code", "wastage", "usage", "b.o.m", "bom", "final rate"]

def _boq_sheet_score(header_row, row_count):
    """Score a worksheet for BOQ suitability — higher = more likely the main BOQ sheet."""
    score = 0
    non_null = [h for h in header_row if h is not None and str(h).strip()]
    if len(non_null) < 2:
        return -999
    for h in header_row:
        if not h:
            continue
        hl = _norm_header(str(h))
        # Heavy penalty for resource-analysis column headers
        if any(kw in hl for kw in _RESOURCE_SHEET_PENALTIES):
            score -= 30
            continue
        # Reward matching any known BOQ field keyword
        for _field, keywords in _BOQ_KEYWORD_RULES:
            if any(_norm_header(kw) in hl for kw in keywords):
                score += 15
                break
        else:
            # Small bonus for Arabic content even if no keyword match
            if any('؀' <= c <= 'ۿ' for c in str(h)):
                score += 2
    # Penalise extremely large sheets (resource breakdowns can have 5 000+ rows)
    if row_count and row_count > 1200:
        score -= 20
    return score

def _keyword_map_boq(headers):
    """Map column indices to BOQ field names using Arabic/English keywords."""
    result = {}
    claimed_fields = set()
    for i, h in enumerate(headers):
        hl = _norm_header(h)
        if not hl or any(kw in hl for kw in _BOQ_SKIP_KEYWORDS):
            continue
        # Skip auto-generated no-header names like Col0, Col1, Col2 ...
        if re.match(r'^col\d+$', hl):
            continue
        for field, keywords in _BOQ_KEYWORD_RULES:
            if field in claimed_fields:
                continue
            if any(_norm_header(kw) in hl for kw in keywords):
                result[str(i)] = field
                claimed_fields.add(field)
                break
    logging.info("Keyword BOQ map: %s", result)
    return result
def _parse_boq_impl():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    _boq_override = request.form.get("api_key", "").strip() or None

    try:
        import openpyxl
        raw_bytes = f.read()
        # Single open — use max_row metadata (instant, no row iteration) to pick sheet
        wb = openpyxl.load_workbook(io.BytesIO(raw_bytes), read_only=True, data_only=True)
        best_ws   = wb.active.title if wb.active else (wb.sheetnames[0] if wb.sheetnames else None)
        best_score = -999

        for sname in wb.sheetnames:
            ws_try    = wb[sname]
            row_count = ws_try.max_row or 0
            if row_count < 2:
                continue
            # Read first ≤10 rows to find the header and score the sheet
            header_row = None
            sampled = 0
            for row_vals in ws_try.iter_rows(min_row=1, max_row=10, values_only=True):
                sampled += 1
                non_empty = [c for c in row_vals if c is not None and str(c).strip()]
                if len(non_empty) >= 3:
                    header_row = row_vals
                    break
            if header_row is None:
                continue
            score = _boq_sheet_score(header_row, row_count)
            logging.info("Sheet '%s': score=%d, rows=%d", sname, score, row_count)
            if score > best_score:
                best_score, best_ws = score, sname

        ws = wb[best_ws] if best_ws else wb.active
        if ws is None:
            wb.close()
            return jsonify({"error": "Could not read any sheet from the Excel file"}), 400
        logging.info("Selected sheet '%s' (score=%d)", best_ws, best_score)
        # Read up to 600 rows (header search + up to 300 data rows with buffer)
        raw_rows = []
        for row in ws.iter_rows(values_only=True):
            raw_rows.append(row)
            if len(raw_rows) >= 600:
                break
        wb.close()
    except Exception as e:
        return jsonify({"error": f"Failed to read Excel file: {e}"}), 400

    if len(raw_rows) < 2:
        return jsonify({"error": "Excel file is empty or has no data"}), 400

    # Find first row that has at least 3 non-empty cells → use as headers
    headers = []
    data_start = 0
    for i, row in enumerate(raw_rows):
        non_empty = [c for c in row if c is not None and str(c).strip()]
        if len(non_empty) >= 3:
            headers = [str(c).strip() if c is not None else f"Col{j}" for j, c in enumerate(row)]
            data_start = i + 1
            break

    if not headers:
        return jsonify({"error": "Could not detect a header row (need ≥3 non-empty cells)"}), 400

    # Collect up to 300 data rows (skip fully empty rows)
    data_rows = []
    for row in raw_rows[data_start: data_start + 300]:
        values = [str(c).strip() if c is not None else "" for c in row]
        if any(v for v in values):
            data_rows.append(values)

    if not data_rows:
        return jsonify({"error": "No data rows found after the header"}), 400

    # Step 1: keyword-based mapping (works offline, handles Arabic + English)
    column_map = _keyword_map_boq(headers)

    # Step 2: AI mapping (enhances/overrides keyword map if AI is available)
    sample = data_rows[:3]
    prompt = (
        "You are mapping an Excel BOQ (Bill of Quantities) file to Odoo construction models.\n\n"
        "EXCEL COLUMNS (index: header | sample values):\n"
        + "\n".join(
            f"  {i}: {h} | {' / '.join(str(r[i]) for r in sample if i < len(r) and r[i])}"
            for i, h in enumerate(headers)
        )
        + "\n\nAVAILABLE ODOO FIELDS (map each column to exactly one):\n"
        "  COMMON (all BOQ models):\n"
        "    name          = البند / Description / item description (REQUIRED — must map this)\n"
        "    quantity      = الكمية / Quantity / Qty\n"
        "    unit_cost     = الفئة / Unit Cost / Rate / Unit Price\n"
        "    total_cost    = الإجمالي / Total / Amount\n"
        "    item_code     = القسم / رقم البند / Line Code / Item No\n"
        "    unit          = الوحدة / UOM / Unit of Measure / Description UOM\n"
        "    work_type     = نوع العمل / Category / Description Categ\n"
        "    notes         = ملاحظات / Remarks / Notes\n"
        "  project.main.item.line (chapter/section headers with no qty):\n"
        "    section_name  = Chapter name / Main Item Line (text, not ID)\n"
        "  project.detailed.item.line:\n"
        "    main_item_line_id = Main Item Line ID (INTEGER — Odoo record ID e.g. 537)\n"
        "    boq_item_id       = Description ID / BOQ Item ID (INTEGER)\n"
        "  project.subcontracting.boq.line:\n"
        "    boq_contract_id   = BOQ Contract ID (INTEGER)\n\n"
        "MAPPING RULES:\n"
        "  - البند/Description (not ID) → name\n"
        "  - الكمية/Quantity/Qty → quantity\n"
        "  - الوحدة/UOM/Description UOM → unit\n"
        "  - الفئة/Rate/Unit Cost/Unit Price → unit_cost\n"
        "  - الإجمالي/Total/Amount → total_cost\n"
        "  - القسم/Line Code/Item No → item_code\n"
        "  - 'Main Item Line ID' column with integers (537,538...) → main_item_line_id\n"
        "  - 'Main Item Line' column with text only → section_name\n"
        "  - 'Description ID' / 'Description Categ' → boq_item_id / work_type\n"
        "  - 'ITEM' / 'ITEMS' / 'ITEM DESCRIPTION' standalone column with text → name (NOT section_name)\n"
        "  - 'DIVISION' / 'TRADE' / 'DISCIPLINE' → work_type\n"
        "  - 'BOQ RATE' / 'BOQ UNIT RATE' → unit_cost\n"
        "  - 'BOQ AMOUNT' / 'BOQ TOTAL' → total_cost\n"
        "  - 'BOQ UNIT' / 'BOQ UOM' → unit\n"
        "  - Columns named 'Col0', 'Col1', 'Col2' (no header text) → SKIP or item_code for first one\n"
        "  - SKIP: #N/A columns, Cost Code, Wastage, Usage, Resource Code, billed, remain\n\n"
        "CRITICAL RULE: section_name is ONLY for chapter/section HEADERS that have NO quantity and NO price.\n"
        "If a column has descriptions AND quantities exist in the same rows → map it to name.\n\n"
        "Return ONLY a JSON object: {\"col_index\": \"field_name\", ...}\n"
        "Example: {\"1\":\"item_code\",\"2\":\"name\",\"3\":\"quantity\",\"4\":\"unit\",\"5\":\"unit_cost\",\"6\":\"total_cost\"}"
    )
    try:
        _boq_client, _boq_model = _make_chat_client(_boq_override)
        if _boq_client is not None:
            resp = _boq_client.chat.completions.create(
                model=_boq_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300, temperature=0,
                timeout=25,
            )
            raw_json = resp.choices[0].message.content.strip()
            # extract JSON — allow multiline
            m = re.search(r'\{[\s\S]*?\}', raw_json)
            if m:
                ai_map = json.loads(m.group())
                # AI map wins over keyword map
                column_map.update(ai_map)
                logging.info("AI column map: %s", ai_map)
    except Exception as e:
        logging.warning("AI column mapping failed (using keyword map): %s", e)

    return jsonify({
        "headers": headers,
        "rows": data_rows,
        "column_map": column_map,
        "total_rows": len(data_rows),
    })


@app.route("/get-projects", methods=["GET"])
@app.route("/ai/get-projects", methods=["GET"])
def get_projects():
    """Return list of projects from Odoo for the project selector."""
    try:
        projects = odoo_call("project.project", "search_read", [[]], {"fields": ["id", "name"], "limit": 100, "order": "name asc"})
        return jsonify(projects)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/import-boq", methods=["POST"])
@app.route("/ai/import-boq", methods=["POST"])
def import_boq():
    """Import mapped BOQ rows into Odoo models."""
    data       = request.json
    rows       = data.get("rows", [])
    column_map = data.get("column_map", {})   # {"col_index_str": "field_name"}
    project_id = data.get("project_id")
    models_to_import = data.get("models", ["project.subcontracting.boq.line", "project.detailed.item.line"])

    if not rows:
        return jsonify({"error": "No rows to import"}), 400
    if not column_map:
        return jsonify({"error": "Column mapping is required"}), 400
    if not project_id:
        return jsonify({"error": "Project is required"}), 400

    col_map = {int(k): v for k, v in column_map.items()}

    def _to_float(s):
        try:
            return float(str(s).replace(",", "").replace(" ", "") or 0)
        except Exception:
            return 0.0

    def _is_section_header(vals, qty, unit_cost):
        """True when a row is a chapter/section header (no numeric values)."""
        if qty != 0 or unit_cost != 0:
            return False
        code = str(vals.get("item_code", "")).strip()
        # Section codes look like "1", "2", "A", "I" — no dash or dot
        if code and '-' not in code and '.' not in code:
            return True
        # Or: no code at all but name is short (chapter title)
        if not code and vals.get("name") and len(vals["name"]) < 80:
            return True
        return False

    imported        = 0
    sections        = 0
    skipped         = 0
    errors          = []
    boq_contract_id = None
    current_main_id = None   # project.main.item.line id of current section
    last_name       = None   # carry-forward for merged-cell continuation rows
    last_item_code  = None

    def _err(e):
        s = getattr(e, "faultString", None) or str(e)
        return s.replace("<", "[").replace(">", "]")[:500]

    def _to_int_id(v):
        """Convert a cell value like '537' or '537.0' to int, or None."""
        try:
            return int(float(str(v).replace(",", "").strip()))
        except Exception:
            return None

    # ── Find existing boq.contract for the project (needed for subcontracting) ──
    if "project.subcontracting.boq.line" in models_to_import:
        try:
            existing_contracts = odoo_call("boq.contract", "search_read",
                [[["project_id", "=", project_id]]],
                {"fields": ["id", "name"], "limit": 1, "order": "id desc"})
            if existing_contracts:
                boq_contract_id = existing_contracts[0]["id"]
                logging.info("Using existing boq.contract id=%d '%s'",
                             boq_contract_id, existing_contracts[0]["name"])
            else:
                # Try creating a minimal contract
                from datetime import date as _date
                try:
                    boq_contract_id = odoo_call("boq.contract", "create",
                        [{"name": f"Tender Import {_date.today()}", "project_id": project_id}])
                except Exception as ce:
                    logging.warning("boq.contract create failed: %s — subcontracting lines will be skipped", ce)
                    models_to_import = [m for m in models_to_import
                                        if m != "project.subcontracting.boq.line"]
        except Exception as e:
            logging.warning("boq.contract lookup failed: %s", e)

    sub_ok = det_ok = 0

    for i, row in enumerate(rows):
        vals = {}
        for col_idx, field in col_map.items():
            if col_idx < len(row) and row[col_idx]:
                vals[field] = row[col_idx]

        # Pre-compute numeric values (needed for section vs item detection below)
        qty_pre       = _to_float(vals.get("quantity", 0))
        uc_pre        = _to_float(vals.get("unit_cost") or vals.get("unit_price", 0))
        tc_pre        = _to_float(vals.get("total_cost", 0))

        # ── section_name check FIRST ────────────────────────────────────────────
        # This must come before the name-skip so rows that only have section_name
        # are not silently dropped.
        explicit_sec = str(vals.get("section_name", "")).strip()
        if explicit_sec:
            if qty_pre or uc_pre or tc_pre:
                # Has quantities → section_name column was mis-mapped; treat as item name
                if not vals.get("name"):
                    vals["name"] = explicit_sec
                # Fall through to normal item processing below
            else:
                # No quantities → true section / chapter header
                try:
                    sec_id = odoo_call("project.main.item.line", "create",
                        [{"name": explicit_sec, "project_id": project_id}])
                    current_main_id = sec_id
                    sections += 1
                    last_name = explicit_sec
                except Exception as e:
                    errors.append(f"Row {i+1} section: {_err(e)}")
                continue

        # ── Carry-forward: merged-cell rows have qty/price but no name ──────────
        if not vals.get("name"):
            if last_name and (qty_pre or uc_pre or tc_pre):
                vals["name"] = last_name
                if last_item_code and not vals.get("item_code"):
                    vals["item_code"] = last_item_code

        if not vals.get("name"):
            skipped += 1
            continue

        # Update carry-forward state
        last_name = vals["name"]
        if vals.get("item_code"):
            last_item_code = vals["item_code"]

        qty        = _to_float(vals.get("quantity", 0))
        unit_cost  = _to_float(vals.get("unit_cost") or vals.get("unit_price", 0))
        total_cost = _to_float(vals.get("total_cost", 0))
        if qty and unit_cost and not total_cost:
            total_cost = qty * unit_cost
        elif qty and total_cost and not unit_cost:
            unit_cost = round(total_cost / qty, 4)

        # Resolve Many2one IDs provided directly in the file
        file_main_id  = _to_int_id(vals.get("main_item_line_id"))
        file_boq_id   = _to_int_id(vals.get("boq_item_id"))
        row_main_id   = file_main_id or current_main_id  # file ID wins, else auto-tracked

        # ── Auto-detect section header (no qty, short code without dash) ────
        if _is_section_header(vals, qty, unit_cost) and not file_main_id:
            code = str(vals.get("item_code", "")).strip()
            sec_name = f"[{code}] {vals['name']}" if code else vals["name"]
            try:
                sec_id = odoo_call("project.main.item.line", "create",
                    [{"name": sec_name, "project_id": project_id}])
                current_main_id = sec_id
                sections += 1
            except Exception as e:
                errors.append(f"Row {i+1} section: {_err(e)}")
            continue

        # Build display name
        item_name = vals["name"]
        if vals.get("item_code"):
            item_name = f"[{vals['item_code']}] {vals['name']}"

        # ── project.subcontracting.boq.line ──────────────────────────────────
        if "project.subcontracting.boq.line" in models_to_import:
            try:
                rec = {"name": item_name, "project_id": project_id}
                if boq_contract_id: rec["boq_contract_id"] = boq_contract_id
                if qty:             rec["quantity"]  = qty
                if unit_cost:       rec["unit_cost"] = unit_cost
                if vals.get("work_type"): rec["work_type"] = vals["work_type"]
                odoo_call("project.subcontracting.boq.line", "create", [rec])
                sub_ok += 1
            except Exception as e:
                errors.append(f"Row {i+1} subcontracting: {_err(e)}")
                logging.warning("subcontracting row %d: %s", i+1, e)

        # ── project.detailed.item.line ────────────────────────────────────────
        if "project.detailed.item.line" in models_to_import:
            try:
                rec2 = {"name": item_name, "project_id": project_id}
                if qty:          rec2["quantity"]  = qty
                if unit_cost:    rec2["unit_cost"] = unit_cost
                if row_main_id:  rec2["main_item_line_id"] = row_main_id
                if file_boq_id:  rec2["boq_item_id"] = file_boq_id
                odoo_call("project.detailed.item.line", "create", [rec2])
                det_ok += 1
            except Exception as e:
                err_msg = _err(e)
                # FK violation on boq_item_id → retry without it (file value not in DB)
                if file_boq_id and ("boq_item_id" in err_msg or "fkey" in err_msg.lower()):
                    try:
                        rec2.pop("boq_item_id", None)
                        odoo_call("project.detailed.item.line", "create", [rec2])
                        det_ok += 1
                        logging.info("Row %d: imported without boq_item_id (FK miss)", i+1)
                    except Exception as e2:
                        errors.append(f"Row {i+1} detailed: {_err(e2)}")
                        logging.warning("detailed row %d (retry): %s", i+1, e2)
                else:
                    errors.append(f"Row {i+1} detailed: {err_msg}")
                    logging.warning("detailed row %d: %s", i+1, e)

        if len(errors) >= 50:
            errors.append("...import stopped after 50 errors — check Railway logs")
            break

    imported = max(sub_ok, det_ok)

    return jsonify({
        "imported":         imported,
        "sub_ok":           sub_ok,
        "det_ok":           det_ok,
        "sections":         sections,
        "skipped":          skipped,
        "errors":           errors,
        "boq_contract_id":  boq_contract_id,
        "models_used":      list(models_to_import) + (
            ["project.main.item.line"] if sections > 0 else []),
    })


# ── WhatsApp (Twilio Sandbox) ──────────────────────────────────────────────────

def _twilio_send(to_number, body):
    """Send a WhatsApp message via Twilio REST API."""
    sid   = os.environ.get("TWILIO_ACCOUNT_SID", "")
    token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    from_ = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    # Ensure whatsapp: prefix is present
    if from_ and not from_.startswith("whatsapp:"):
        from_ = "whatsapp:" + from_
    if not to_number.startswith("whatsapp:"):
        to_number = "whatsapp:" + to_number.lstrip("whatsapp:")
    if not sid or not token:
        logging.warning("Twilio credentials not set — cannot send WhatsApp reply")
        return
    if not _requests:
        logging.warning("requests library not installed")
        return
    auth = base64.b64encode(f"{sid}:{token}".encode()).decode()
    url  = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    try:
        r = _requests.post(url,
            headers={"Authorization": f"Basic {auth}"},
            data={"From": from_, "To": to_number, "Body": body},
            timeout=15)
        if r.status_code >= 400:
            logging.error("Twilio API error %s: %s", r.status_code, r.text[:300])
    except Exception as e:
        logging.error("Twilio send error: %s", e)


def _md_to_whatsapp(text):
    """Convert markdown to WhatsApp-compatible formatting."""
    # **bold** → *bold*
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    # # Heading → *Heading*
    text = re.sub(r'^#{1,4}\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)
    # Remove code fences
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    # Strip chart data lines
    text = re.sub(r'CHART_\w+:\{[^\n]+\}', '', text)
    # Collapse excess blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _split_whatsapp(text, max_len=4000):
    """Split message at paragraph boundaries so each chunk ≤ max_len chars."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while len(text) > max_len:
        pos = text.rfind('\n\n', 0, max_len)
        if pos == -1:
            pos = text.rfind('\n', 0, max_len)
        if pos == -1:
            pos = max_len
        chunks.append(text[:pos].strip())
        text = text[pos:].strip()
    if text:
        chunks.append(text)
    return chunks


def _run_ai_sync(user_text):
    """Run the full AI + Odoo tool-call loop synchronously. Returns answer text."""
    client, _model = _make_chat_client()
    if not client:
        return "❌ AI API key not configured on the server."

    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user",   "content": user_text},
    ]

    for _ in range(3):
        _has_results = any(m.get("role") == "tool" for m in messages)
        call_kw = dict(
            model=_model, messages=messages, tools=TOOLS,
            tool_choice="none" if _has_results else "auto",
            max_tokens=1800, temperature=0.1,
        )
        try:
            resp = client.chat.completions.create(**call_kw)
        except Exception as e:
            return f"❌ AI error: {str(e)[:200]}"

        msg    = resp.choices[0].message
        finish = resp.choices[0].finish_reason

        asst: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            asst["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(asst)

        if finish in ("stop", "end_turn") or not msg.tool_calls:
            return msg.content or "لم أستطع الإجابة على هذا السؤال."

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            result = run_tool(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "لم أتمكن من إكمال الطلب. يرجى إعادة الصياغة."


def _transcribe_voice(media_url):
    """Download voice note from Twilio and transcribe with Groq Whisper."""
    groq_key = DEFAULT_GROQ_KEY
    if not groq_key or not _requests:
        return None

    sid   = os.environ.get("TWILIO_ACCOUNT_SID", "")
    token = os.environ.get("TWILIO_AUTH_TOKEN", "")

    # Download audio (Twilio requires auth to access media URLs)
    auth = base64.b64encode(f"{sid}:{token}".encode()).decode()
    resp = _requests.get(media_url,
                         headers={"Authorization": f"Basic {auth}"},
                         timeout=30)
    if resp.status_code != 200:
        logging.warning("Failed to download voice note: %s", resp.status_code)
        return None

    audio_bytes = resp.content
    content_type = resp.headers.get("Content-Type", "audio/ogg")
    # Derive a file extension from content type
    ext = "ogg"
    if "mpeg" in content_type or "mp3" in content_type:
        ext = "mp3"
    elif "mp4" in content_type or "m4a" in content_type:
        ext = "mp4"
    elif "wav" in content_type:
        ext = "wav"

    try:
        client_tmp = Groq(api_key=groq_key)
        transcript = client_tmp.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=(f"voice.{ext}", audio_bytes),
            response_format="text",
        )
        return str(transcript).strip()
    except Exception as e:
        logging.error("Whisper transcription failed: %s", e)
        return None


def _handle_whatsapp(from_number, user_text):
    """Background worker: run AI query then send WhatsApp reply."""
    logging.info("WhatsApp from %s: %s", from_number, user_text[:80])
    try:
        answer = _run_ai_sync(user_text)
        formatted = _md_to_whatsapp(answer)
        for chunk in _split_whatsapp(formatted):
            _twilio_send(from_number, chunk)
    except Exception as e:
        _twilio_send(from_number, f"❌ خطأ: {str(e)[:200]}")


@app.route("/whatsapp", methods=["POST"])
@app.route("/ai/whatsapp", methods=["POST"])
def whatsapp_webhook():
    """Twilio WhatsApp webhook — returns TwiML ack immediately, AI reply via REST API."""
    from_number  = request.form.get("From", "").strip()
    body         = request.form.get("Body", "").strip()
    num_media    = int(request.form.get("NumMedia", "0") or "0")
    media_url    = request.form.get("MediaUrl0", "").strip()
    media_type   = request.form.get("MediaContentType0", "").strip()

    logging.info("WA webhook: from=%s body=%r media=%s", from_number, body[:60], media_type)

    def _twiml(msg=None):
        if msg:
            safe = msg.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
            xml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{safe}</Message></Response>'
        else:
            xml = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'
        return (xml, 200, {"Content-Type": "text/xml"})

    if not from_number:
        return _twiml()

    # ── Voice note ──────────────────────────────────────────────────────────
    if num_media > 0 and media_url and "audio" in media_type:
        def _handle_voice():
            try:
                transcript = _transcribe_voice(media_url)
                if not transcript:
                    _twilio_send(from_number, "❌ لم أتمكن من تفريغ الرسالة الصوتية. حاول مرة أخرى.")
                    return
                _twilio_send(from_number, f"🎙️ فهمت: {transcript}")
                answer = _run_ai_sync(transcript)
                for chunk in _split_whatsapp(_md_to_whatsapp(answer)):
                    _twilio_send(from_number, chunk)
            except Exception as e:
                logging.error("WA voice handler error: %s", e)
                _twilio_send(from_number, f"❌ خطأ: {str(e)[:200]}")
        threading.Thread(target=_handle_voice, daemon=True).start()
        # Immediate TwiML ack so Twilio doesn't retry
        return _twiml("🎤 جاري تفريغ الصوت وتحليل سؤالك…")

    # ── Text message ────────────────────────────────────────────────────────
    elif body:
        threading.Thread(
            target=_handle_whatsapp,
            args=(from_number, body),
            daemon=True
        ).start()
        # Immediate TwiML ack — user sees this in ~1 second, full answer follows
        return _twiml("⏳ جاري التحليل والاستعلام من Odoo…")

    return _twiml()


@app.route("/wa-test", methods=["GET"])
def wa_test():
    """Diagnostic: check Twilio credentials and optionally send a test message."""
    sid   = os.environ.get("TWILIO_ACCOUNT_SID", "")
    token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    from_ = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    gemini = bool(GEMINI_API_KEY)
    groq  = bool(DEFAULT_GROQ_KEY)
    odoo  = bool(ODOO_API_KEY)
    to_   = request.args.get("to", "")   # ?to=whatsapp:+20XXXXXXXXX

    result = {
        "TWILIO_ACCOUNT_SID":   sid[:8] + "..." if sid else "❌ NOT SET",
        "TWILIO_AUTH_TOKEN":    "✅ set" if token else "❌ NOT SET",
        "TWILIO_WHATSAPP_FROM": from_ or "❌ NOT SET",
        "GEMINI_API_KEY":        "✅ set" if gemini else "❌ NOT SET",
        "GROQ_API_KEY (Whisper)": "✅ set" if groq else "⚠️ not set (voice only)",
        "ODOO_API_KEY":         "✅ set" if odoo else "❌ NOT SET",
        "webhook_url":          request.host_url.rstrip("/") + "/whatsapp",
    }

    if to_ and sid and token and _requests:
        if not from_.startswith("whatsapp:"):
            from_ = "whatsapp:" + from_
        if not to_.startswith("whatsapp:"):
            to_ = "whatsapp:" + to_
        auth = base64.b64encode(f"{sid}:{token}".encode()).decode()
        url  = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
        try:
            r = _requests.post(url,
                headers={"Authorization": f"Basic {auth}"},
                data={"From": from_, "To": to_, "Body": "🤖 Test from Odoo AI Assistant — connection OK!"},
                timeout=15)
            result["test_send"] = {"status": r.status_code, "body": r.text[:400]}
        except Exception as e:
            result["test_send"] = {"error": str(e)}
    elif to_:
        result["test_send"] = "❌ Cannot send — missing Twilio credentials or requests library"

    return jsonify(result)


@app.route("/whatsapp", methods=["GET"])
def whatsapp_info():
    return jsonify({"status": "ok", "webhook": "POST /whatsapp", "hint": "This endpoint accepts POST from Twilio only."})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
