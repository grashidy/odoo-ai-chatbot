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
AI_MODEL        = "gemini-2.0-flash"

# Groq key — only used for Whisper audio transcription
_key_file = Path(__file__).parent / ".groq_key"
DEFAULT_GROQ_KEY = (
    os.environ.get("GROQ_API_KEY", "").strip()
    or (_key_file.read_text(encoding="utf-8").strip() if _key_file.exists() else "")
)

def _make_chat_client(override_key=None):
    """Return (OpenAI-compatible client, model_name) for chat."""
    key = override_key or GEMINI_API_KEY or DEFAULT_GROQ_KEY
    if not key:
        return None, None
    if GEMINI_API_KEY and override_key in (None, GEMINI_API_KEY):
        return OpenAI(api_key=key, base_url=GEMINI_BASE_URL), AI_MODEL
    # Fallback to Groq-compatible if no Gemini key
    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1"), "llama-3.1-8b-instant"

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

_MAX_RESULT_CHARS = 3500  # truncate giant tool results so Groq doesn't choke

def _truncate_result(text):
    if len(text) > _MAX_RESULT_CHARS:
        return text[:_MAX_RESULT_CHARS] + f'\n... [truncated — {len(text)} chars total, showing first {_MAX_RESULT_CHARS}]'
    return text

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
            return _truncate_result(json.dumps(results, ensure_ascii=False, default=str))

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
                        item[k] = v
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
SYSTEM_PROMPT = """You are an AI assistant for a multi-company Odoo 18 system covering Real Estate, Construction, Purchases, and HR.
Reply in the same language as the user (Arabic or English).

CRITICAL RULES:
- NEVER say "I don't have access". You have FULL access to ALL Odoo models.
- Use tools for real data — never guess or fabricate numbers.
- Counts/stats/totals → odoo_read_group. Records/lists → odoo_search (limit 30).
- Format numbers with commas. Currency = EGP. Use markdown tables.
- On field error → call odoo_get_fields once, then retry.
- Unknown model → call odoo_get_fields on it, then query it.
- Multi-company context is injected automatically — do not add it yourself.
- BATCH TOOLS: Call ALL tools you need in ONE single response — never chain tools across multiple turns. If you need project data AND item costs, call both tools simultaneously in one response.
- ONE QUERY RULE: Query each model at most ONCE per user question. Do NOT repeat the same model query. If you already have data, answer immediately.
- AFTER TOOLS: Once you receive tool results, answer immediately. Do NOT call more tools.

CHARTS (include when showing statistics):
CHART_BAR:{"title":"T","labels":["A","B"],"data":[10,20]}
CHART_PIE:{"title":"T","labels":["A","B"],"data":[10,20]}

══════════════════════════════════════════════════════════════
CONSTRUCTION MODELS (custom beyond_fm module):
══════════════════════════════════════════════════════════════

CLIENT-FACING BOQ — what the project owner is billed for:
  project.detailed.item.line
    name, project_id, boq_item_id, main_item_line_id,
    quantity, unit_cost, uom_id, is_subcontracting, is_subcontracting_boq

  project.main.item.line  (project work categories: Structural/MEP/Excavation)
    project_id, main_item_id, name

  project.main.item  (global master catalogue of work types — 48 records, no project_id)
    name, code

  project.boq.item  (global BOQ item catalogue — no project_id)
    name, code, uom_id, billing_type (billable / unbillable)

VENDOR/SUBCONTRACTOR BOQ — what subcontractors execute:
  project.subcontracting.boq.line
    name, project_id, boq_contract_id,
    quantity, unit_cost, boq_cost, billed_qty, remain_qty,
    boq_item_id, boq_item_lines_id (FK → project.detailed.item.line),
    main_item_lines_id (FK → project.main.item.line),
    is_subcontracting_boq, work_type

  boq.contract  (groups sub-BOQ lines into a package — NO financial fields)
    name, project_id, partner_id, state

  subcontractor.contract  (actual signed vendor contract with value)
    name, project_id, partner_id, status,
    bills_amount_total, bills_amount_due,
    total_adv_amount, total_deductions

  subcontractor.contract.line
    name, contract_id, project_id, work_type,
    quantity, billed_qty, remain_qty, assigned_qty,
    unit_price, total_price,
    install, supply, transportation, labor, misc

ADVANCE PAYMENTS:
  construction.advance.payment
    name, partner_id, amount, date, state,
    project_id, due_amount, settled_amount,
    subcontractor_contract_id

CONSTRUCTION PLANS (milestones & operations):
  main.project.plan.operation  (high-level phase milestones)
    name, project_id, date_start, date_end, progress, state

  sub.project.plan.operation  (granular activity-level operations)
    name, main_plan_id, date_start, date_end, progress, state

  project.plan.operation  (general plan operation)
    name, project_id, state, progress

══════════════════════════════════════════════════════════════
MODEL ROUTING — always pick the correct model:
══════════════════════════════════════════════════════════════

"advance payments / مدفوعات مقدمة / دفعات مقدمة / مقدمات للمقاولين"
  → construction.advance.payment  (ALWAYS — NEVER use subcontractor.contract for payments)

"contract value / قيمة العقود / عقود المقاولين"
  → subcontractor.contract  (bills_amount_total, bills_amount_due)
  → NEVER use boq.contract for financial questions

"client-facing BOQ / مقايسة المشروع / ما يُفاتَر به العميل / بنود الأعمال للمالك"
  → project.detailed.item.line

"subcontracting BOQ / بنود عقد الباطن / ما ينفذه المقاول / بنود التنفيذ"
  → project.subcontracting.boq.line

"BOQ contract grouping / تجميع عقود المقاولة"
  → boq.contract  (structure only, no money fields)

"construction milestones / خطة المشروع / مراحل التنفيذ / project plan"
  → main.project.plan.operation (high-level) + sub.project.plan.operation (detail)

"contract line items / بنود العقد / unit price per item"
  → subcontractor.contract.line

"BOQ items catalogue / كتالوج بنود المقايسة"
  → project.boq.item  (global — no project_id filter needed)

══════════════════════════════════════════════════════════════
REAL ESTATE MODELS (project.rs.* prefix — CRITICAL):
══════════════════════════════════════════════════════════════
project.rs.unit     → name, state (available/reserved/contracted/booked/delivered/cancelled),
                       project_id, net_area, current_sale_price, partner_id,
                       building_id, phase_id, unit_code, floor
project.rs.building → name, project_id, phase_id
project.rs.phase    → name, project_id
project.project     → name, user_id  (all projects — construction and RE)

IMPORTANT: NEVER use rs.unit or rs.project (wrong prefix). Always project.rs.unit, project.rs.building, project.rs.phase.

══════════════════════════════════════════════════════════════
STANDARD ODOO MODELS:
══════════════════════════════════════════════════════════════
purchase.order      → name, partner_id, state, amount_total, date_order, project_id
                      state: 'draft'=RFQ, 'purchase'=Confirmed, 'done'=Received
purchase.order.line → order_id, product_id, product_qty, price_unit, price_subtotal
hr.employee         → name, department_id, job_title, job_id, work_phone, mobile_phone, active
hr.department       → name, manager_id
project.task        → name, project_id, stage_id, date_deadline, kanban_state, user_ids
res.partner         → name, phone, mobile, email, is_company

══════════════════════════════════════════════════════════════
ARABIC → MODEL MAPPING:
══════════════════════════════════════════════════════════════
"مدفوعات مقدمة / دفعات مقدمة / مقدمات"        → construction.advance.payment
"عقود مقاولة / قيمة عقد / عقد مقاول"          → subcontractor.contract
"بنود مقايسة العميل / BOQ المشروع"            → project.detailed.item.line
"بنود الباطن / بنود التنفيذ / بنود عقد الباطن" → project.subcontracting.boq.line
"خطة المشروع / مراحل التنفيذ / الخطة الزمنية"  → main.project.plan.operation
"وحدات عقارية / الوحدات"                      → project.rs.unit
"مباني / مرحلة عقارية"                        → project.rs.building / project.rs.phase
"موظفين / employees"                          → hr.employee
"مشاريع"                                      → project.project
"طلبات الشراء / مشتريات"                      → purchase.order
"كتالوج بنود المقايسة"                        → project.boq.item

══════════════════════════════════════════════════════════════
QUERY EXAMPLES:
══════════════════════════════════════════════════════════════
Advance payments per contractor:
  odoo_read_group model="construction.advance.payment" domain=[] groupby=["partner_id"] aggregates=["amount:sum","due_amount:sum"]

Client BOQ lines for a project:
  odoo_search model="project.detailed.item.line" domain=[["project_id","=",24]] fields=["name","boq_item_id","quantity","unit_cost","uom_id","main_item_line_id"]

Subcontract BOQ lines (vendor scope, is_subcontracting_boq=True):
  odoo_search model="project.subcontracting.boq.line" domain=[["is_subcontracting_boq","=",true]] fields=["name","boq_contract_id","quantity","unit_cost","boq_item_lines_id","billed_qty"]

Subcontractor contracts with advance payments balance:
  odoo_read_group model="subcontractor.contract" domain=[] groupby=["project_id","partner_id"] aggregates=["bills_amount_total:sum","total_adv_amount:sum"]

Construction plan milestones:
  odoo_search model="main.project.plan.operation" domain=[] fields=["name","project_id","date_start","date_end","progress","state"]

RE units available with prices:
  odoo_search model="project.rs.unit" domain=[["state","=","available"]] fields=["name","current_sale_price","net_area","project_id","building_id"]

RE units by state (chart-ready):
  odoo_read_group model="project.rs.unit" domain=[] groupby=["state"] aggregates=[]

Confirmed POs by vendor:
  odoo_read_group model="purchase.order" domain=[["state","in",["purchase","done"]]] groupby=["partner_id"] aggregates=["amount_total:sum"]

Main work categories with BOQ value per project:
  odoo_read_group model="project.detailed.item.line" domain=[] groupby=["project_id","main_item_line_id"] aggregates=["unit_cost:sum"]

Overdue construction milestones:
  odoo_search model="main.project.plan.operation" domain=[["state","!=","done"]] fields=["name","project_id","date_end","progress"] order="date_end asc"

Blocked tasks:
  odoo_search model="project.task" domain=[["kanban_state","=","blocked"]] fields=["name","project_id","stage_id","date_deadline","user_ids"]

BOQ items catalogue (global, billable):
  odoo_search model="project.boq.item" domain=[["billing_type","=","billable"]] fields=["name","code","uom_id"]

COUNTING: odoo_read_group with aggregates=[] returns 'count' automatically. NEVER add 'id:count'.
FIELD COMPARISONS: Odoo domain cannot compare two fields directly. For billed_qty > quantity: fetch all rows with odoo_search limit=200, then note in your answer which lines exceed the threshold."""

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
            # Keep only last 6 messages from history to limit token usage
            recent = history[-6:] if len(history) > 6 else history
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
                    yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"
                    answered = True
                    break

                # Notify client and count tool calls to detect loops.
                # Key = tool_name:model so querying two DIFFERENT models with
                # odoo_search is NOT treated as a loop.
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {}
                    yield f"data: {json.dumps({'type': 'tool', 'name': tc.function.name, 'input': args})}\n\n"
                    _model_arg = args.get("model", "") if isinstance(args, dict) else ""
                    _call_key  = f"{tc.function.name}:{_model_arg}"
                    tool_call_counts[_call_key] = tool_call_counts.get(_call_key, 0) + 1

                # Loop = same (tool, model) called twice
                loop_detected = any(v >= 2 for v in tool_call_counts.values())

                # Execute all tools and collect results (with per-call timeout)
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {}
                    _tool_result = [None]
                    _tool_done   = threading.Event()
                    def _do_tool(n=tc.function.name, a=args):
                        try:
                            _tool_result[0] = run_tool(n, a)
                        except Exception as _te:
                            _tool_result[0] = json.dumps({"error": str(_te)})
                        finally:
                            _tool_done.set()
                    threading.Thread(target=_do_tool, daemon=True).start()
                    # Wait up to 45 s, sending SSE pings every 5 s so Railway
                    # doesn't drop the idle connection during slow Odoo queries
                    _waited = 0
                    while _waited < 45 and not _tool_done.is_set():
                        _tool_done.wait(timeout=5)
                        _waited += 5
                        if not _tool_done.is_set() and _waited < 45:
                            yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                    if not _tool_done.is_set():
                        _tool_result[0] = json.dumps({"error": "Odoo query timed out after 45s. The server may be busy — try a simpler query or smaller limit."})
                        logging.warning("Tool %s timed out after 45 s", tc.function.name)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": _tool_result[0] or json.dumps({"error": "empty result"})
                    })

                if loop_detected:
                    messages.append({
                        "role": "user",
                        "content": "You have enough data. Stop calling tools and give the final answer now."
                    })
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
    # Construction: BOQ contracts by project
    boq_contracts_by_project = safe("boq.contract", "read_group",
        [[], ["project_id"], ["project_id"]], {"lazy": False})
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
    # Real Estate: units by state — model prefix is project.rs.unit (NOT rs.unit)
    units_by_state = safe("project.rs.unit", "read_group",
        [[], ["state"], ["state"]], {"lazy": False})
    # Construction: detailed BOQ lines by project (client-facing BOQ)
    boq_detail_by_project = safe("project.detailed.item.line", "read_group",
        [[], ["project_id", "unit_cost:sum"], ["project_id"]], {"lazy": False})
    # Construction: plan milestones by state
    plan_by_state = safe("main.project.plan.operation", "read_group",
        [[], ["state"], ["state"]], {"lazy": False})

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
        "boq_by_project":           extract(boq_by_project,           "project_id", amount_field="boq_cost"),
        "boq_contracts_by_project": extract(boq_contracts_by_project, "project_id"),
        "boq_detail_by_project":    extract(boq_detail_by_project,    "project_id", amount_field="unit_cost"),
        "adv_by_state":             extract(adv_by_state,             "state"),
        "adv_by_project":           extract(adv_by_project,           "project_id", amount_field="amount"),
        "items_by_project":         extract(items_by_project,         "project_id", amount_field="total_cost"),
        "po_by_vendor":             extract(po_by_vendor,             "partner_id",  amount_field="amount_total"),
        "emp_by_dept":              extract(emp_by_dept,              "department_id"),
        "units_by_state":           extract(units_by_state,           "state"),
        "plan_by_state":            extract(plan_by_state,            "state"),
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
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    _boq_override = request.form.get("api_key", "").strip() or None

    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(f.read()), read_only=True, data_only=True)
        ws = wb.active
        raw_rows = list(ws.iter_rows(values_only=True))
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

    # Ask Groq to map column headers → Odoo BOQ field names
    column_map = {}
    if groq_key:
        sample = data_rows[:3]
        prompt = (
            "I have an Excel BOQ (Bill of Quantities) tender document.\n"
            f"Column headers (0-indexed): {list(enumerate(headers))}\n"
            f"Sample rows: {sample}\n\n"
            "Map each column INDEX to one of these Odoo field names:\n"
            "  name        → item description / work item name (required)\n"
            "  item_code   → item number / reference code\n"
            "  quantity    → planned/BOQ quantity (numeric)\n"
            "  unit        → unit of measure (m2, m3, kg, ls, etc)\n"
            "  unit_price  → unit rate / unit price (numeric)\n"
            "  work_type   → type of work (supply / install / civil / labor / etc)\n"
            "  notes       → remarks or notes\n\n"
            "Return ONLY valid JSON like: {\"0\": \"item_code\", \"1\": \"name\", \"3\": \"quantity\", \"4\": \"unit_price\"}\n"
            "Skip columns that don't map to any field (totals, subtotals, row numbers)."
        )
        try:
            _boq_client, _boq_model = _make_chat_client(_boq_override)
            resp = _boq_client.chat.completions.create(
                model=_boq_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400, temperature=0,
            )
            raw_json = resp.choices[0].message.content.strip()
            m = re.search(r'\{[^{}]+\}', raw_json, re.DOTALL)
            if m:
                column_map = json.loads(m.group())
        except Exception as e:
            logging.warning("Groq column mapping failed: %s", e)

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

    imported = 0
    skipped  = 0
    errors   = []
    boq_contract_id = None

    # Auto-create a BOQ contract to group all imported lines
    if "project.subcontracting.boq.line" in models_to_import:
        from datetime import date as _date
        contract_name = f"Tender Import {_date.today().strftime('%Y-%m-%d')}"
        try:
            existing = odoo_call("boq.contract", "search_read",
                [[["name", "=", contract_name], ["project_id", "=", project_id]]],
                {"fields": ["id"], "limit": 1})
            if existing:
                boq_contract_id = existing[0]["id"]
            else:
                boq_contract_id = odoo_call("boq.contract", "create",
                    [{"name": contract_name, "project_id": project_id}])
        except Exception as e:
            return jsonify({"error": f"Could not create BOQ contract: {e}"}), 500

    for i, row in enumerate(rows):
        try:
            vals = {}
            for col_idx, field in col_map.items():
                if col_idx < len(row) and row[col_idx]:
                    vals[field] = row[col_idx]

            # Must have at least a name/description
            if not vals.get("name"):
                skipped += 1
                continue

            qty        = _to_float(vals.get("quantity", 0))
            unit_price = _to_float(vals.get("unit_price", 0))

            if "project.subcontracting.boq.line" in models_to_import:
                rec = {
                    "name":             vals["name"],
                    "project_id":       project_id,
                    "boq_contract_id":  boq_contract_id,
                }
                if qty:        rec["quantity"] = qty
                if unit_price: rec["boq_cost"]  = unit_price
                if vals.get("work_type"): rec["work_type"] = vals["work_type"]
                odoo_call("project.subcontracting.boq.line", "create", [rec])

            if "project.detailed.item.line" in models_to_import:
                rec2 = {
                    "name":       vals["name"],
                    "project_id": project_id,
                }
                if qty:        rec2["quantity"]     = qty
                if unit_price: rec2["initial_cost"] = unit_price
                if qty and unit_price:
                    rec2["total_cost"] = qty * unit_price
                odoo_call("project.detailed.item.line", "create", [rec2])

            imported += 1

        except Exception as e:
            errors.append(f"Row {i + 1}: {str(e)[:120]}")
            if len(errors) >= 10:
                break

    return jsonify({
        "imported": imported,
        "skipped":  skipped,
        "errors":   errors,
        "boq_contract_id": boq_contract_id,
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
        {"role": "system", "content": SYSTEM_PROMPT},
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
