import xmlrpc.client, json, os, time, re, logging, threading, io, base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from groq import Groq
try:
    import requests as _requests
except ImportError:
    _requests = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Load Groq key: env var (cloud) → .groq_key file (local) ───────────────────
_key_file = Path(__file__).parent / ".groq_key"
DEFAULT_GROQ_KEY = (
    os.environ.get("GROQ_API_KEY", "").strip()
    or (_key_file.read_text(encoding="utf-8").strip() if _key_file.exists() else "")
)

# ── Odoo connection ────────────────────────────────────────────────────────────
ODOO_URL     = "https://t2-18.odooegypt.com"
ODOO_DB      = "team2_beta_empty"
ODOO_UID     = 2
ODOO_API_KEY = "ab0ae5ad6e2623ca1acd5892e0a07c6a2add8695"

_odoo = xmlrpc.client.ServerProxy(ODOO_URL + "/xmlrpc/2/object")

def odoo_call(model, method, args, kwargs=None):
    return _odoo.execute_kw(ODOO_DB, ODOO_UID, ODOO_API_KEY,
                             model, method, args, kwargs or {})

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

def run_tool(name, args):
    try:
        if name == "odoo_search":
            domain = _coerce_domain(args.get("domain", []))
            limit  = min(int(args.get("limit", 50)), 200)
            raw_fields = args.get("fields", ["display_name"])
            fields = raw_fields if isinstance(raw_fields, list) else _coerce_list(raw_fields)
            if not fields:
                fields = ["display_name"]
            kwargs = {"fields": fields, "limit": limit}
            if args.get("order"):
                kwargs["order"] = args["order"]
            results = odoo_call(args["model"], "search_read", [domain], kwargs)
            return json.dumps(results, ensure_ascii=False, default=str)

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
            # Strip invalid aggregates — Odoo count is automatic (__count), never pass id:count
            agg_fields = [f for f in agg_fields if f not in ("id:count", "id", "__count")]
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
            return json.dumps(cleaned, ensure_ascii=False, default=str)

    except Exception as e:
        return json.dumps({"error": str(e), "hint": "Try odoo_get_fields to check available fields"})

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI assistant for a construction company using Odoo 18.
Reply in same language as user (Arabic or English).

CRITICAL RULES:
- NEVER say "I don't have access" or "I cannot query" any model. You have FULL access to ALL Odoo models. If a model is not in the list below, call odoo_get_fields first to discover its fields, then query it.
- Use tools for real data. Never guess numbers.
- Each tool call at most ONCE per tool. No repeating same tool.
- Counts/stats/totals: use odoo_read_group. Records/lists: use odoo_search (limit 50).
- Format numbers with commas. Prices in EGP. Use markdown tables.
- On field error: call odoo_get_fields once to check fields, then retry.
- Unknown model name? Call odoo_get_fields on it — never refuse.

CHARTS (include when showing statistics):
CHART_BAR:{"title":"T","labels":["A","B"],"data":[10,20]}
CHART_PIE:{"title":"T","labels":["A","B"],"data":[10,20]}

═══════════════════════════════════════════════
MODEL ROUTING — always pick the correct model:
═══════════════════════════════════════════════
"contract value / قيمة العقود / عقود المقاولين"
  → subcontractor.contract  (has bills_amount_total, bills_amount_due)
  → NEVER use boq.contract for financial/value questions

"BOQ lines / بنود المقايسة / supply & install quantities"
  → project.subcontracting.boq.line  (has boq_cost, quantity, billed_qty)

"BOQ structure / مقايسة master record"
  → boq.contract  (structural only, NO financial fields)

"contract line items / بنود العقد / unit price per item"
  → subcontractor.contract.line  (has unit_price, total_price, quantity)

"advance payments / دفعات مقدمة / مقدمات"
  → construction.advance.payment

"detailed progress / تقدم الأعمال / actual vs planned cost"
  → project.detailed.item.line

═══════════════════════════════════════════════
CONSTRUCTION MODELS:
═══════════════════════════════════════════════
subcontractor.contract      → name, date, project_id, partner_id, status,
                               bills_amount_total, bills_amount_due,
                               total_adv_amount, total_deductions
subcontractor.contract.line → name, contract_id, project_id, work_type,
                               quantity, assigned_qty, remain_qty,
                               billed_qty, billed_amount,
                               unit_price, total_price,
                               install, supply, transportation, labor, misc
project.subcontracting.boq.line → name, boq_contract_id, project_id,
                               product_id, quantity, billed_qty,
                               remain_qty, boq_cost, work_type
boq.contract                → name, partner_id, project_id, state
project.detailed.item.line  → name, project_id, quantity, done_qty,
                               initial_cost, actual_cost, total_cost,
                               progress_percentage
construction.advance.payment→ name, partner_id, amount, date, state,
                               project_id, due_amount, settled_amount,
                               subcontractor_contract_id
subcontractor.order         → partner_id, qty, state, production_plan_id
subcontract.boq.progress.line→ name, project_id
project.project             → name, user_id
project.task                → name, project_id, stage_id, date_deadline,
                               kanban_state, user_ids
purchase.order              → name, partner_id, state, amount_total,
                               date_order, project_id
purchase.order.line         → order_id, product_id, product_qty,
                               price_unit, price_subtotal
hr.employee                 → name, department_id, job_title,
                               work_phone, mobile_phone
hr.department               → name, manager_id
res.partner                 → name, phone, mobile, email

REAL ESTATE (ONLY when user explicitly says real estate/عقارات):
rs.project, rs.unit(unit_code,state,rs_project_id,net_area,current_sale_price,partner_id)
rs.contract(partner_id,rs_unit_id,state,contracted_sale_price), rs.installment(partner_id,amount,date,state)

═══════════════════════════════════════════════
QUERY EXAMPLES:
═══════════════════════════════════════════════
Unique subcontractors per project (no duplicates):
  odoo_read_group model="subcontractor.contract" domain=[] groupby=["project_id","partner_id"] aggregates=["bills_amount_total:sum","bills_amount_due:sum","total_adv_amount:sum"]
  → Use this for "اسماء المقاولين / who are the contractors / list contractors per project"
  → groupby TWO fields collapses multiple contracts per subcontractor into one row

Contract value per project (total only):
  odoo_read_group model="subcontractor.contract" domain=[] groupby=["project_id"] aggregates=["bills_amount_total:sum","bills_amount_due:sum"]

List all contracts (flat, may have duplicates per subcontractor):
  odoo_search model="subcontractor.contract" domain=[] fields=["name","partner_id","project_id","status","bills_amount_total","bills_amount_due","total_adv_amount"]

Contract lines (بنود العقد):
  odoo_search model="subcontractor.contract.line" domain=[] fields=["name","contract_id","project_id","quantity","billed_qty","unit_price","total_price","work_type"]

BOQ lines per project:
  odoo_read_group model="project.subcontracting.boq.line" domain=[] groupby=["project_id"] aggregates=["boq_cost:sum","quantity:sum"]

Advance payments per project:
  odoo_read_group model="construction.advance.payment" domain=[] groupby=["project_id"] aggregates=["amount:sum","due_amount:sum"]

Progress per project:
  odoo_read_group model="project.detailed.item.line" domain=[] groupby=["project_id"] aggregates=["total_cost:sum","actual_cost:sum"]

COUNTING: odoo_read_group with aggregates=[] returns 'count' automatically. NEVER add 'id:count' to aggregates.

FIELD-TO-FIELD COMPARISONS: Odoo domain cannot compare two fields. For billed_qty > quantity: use odoo_search domain=[] limit=200, then filter in Python from returned data."""

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
    groq_key = data.get("api_key", "").strip() or DEFAULT_GROQ_KEY

    if not groq_key:
        return jsonify({"error": "Groq API key is required. Click ⚙ and enter your key."}), 400

    client = Groq(api_key=groq_key)

    def generate():
        try:
            # Keep only last 6 messages from history to limit token usage
            recent = history[-6:] if len(history) > 6 else history
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for m in recent:
                messages.append({"role": m["role"], "content": m.get("content") or ""})

            max_iterations   = 5
            tool_call_counts = {}   # tool_name → how many times called total
            tool_fail_count  = 0   # consecutive schema-validation failures
            answered         = False

            for iteration in range(max_iterations):
                response = None
                # Run Groq call in a thread so we can send SSE keepalive pings
                # while waiting — Railway drops idle connections after ~60 s
                _result  = [None]
                _api_err = [None]
                _done    = threading.Event()

                def _groq_call():
                    try:
                        _result[0] = client.chat.completions.create(
                            model="meta-llama/llama-4-scout-17b-16e-instruct",
                            messages=messages,
                            tools=TOOLS,
                            tool_choice="auto",
                            max_tokens=4096,
                            temperature=0.1,
                        )
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
                        # Parse actual wait time from Groq error message
                        m_wait = re.search(r'try again in ([\d.]+)s', err_msg, re.IGNORECASE)
                        wait_sec = int(float(m_wait.group(1))) + 2 if m_wait else 65
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

                # Notify client and count tool calls to detect loops
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {}
                    yield f"data: {json.dumps({'type': 'tool', 'name': tc.function.name, 'input': args})}\n\n"
                    tool_call_counts[tc.function.name] = tool_call_counts.get(tc.function.name, 0) + 1

                # If any single tool has been called 3+ times, stop looping
                loop_detected = any(v >= 3 for v in tool_call_counts.values())

                # Execute all tools and collect results
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {}
                    result = run_tool(tc.function.name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result
                    })

                if loop_detected:
                    messages.append({
                        "role": "user",
                        "content": "You have enough data. Stop calling tools and give the final answer now."
                    })
                    _fr = [None]; _fe = threading.Event()
                    def _final_call():
                        try:
                            _fr[0] = client.chat.completions.create(
                                model="meta-llama/llama-4-scout-17b-16e-instruct",
                                messages=messages,
                                max_tokens=2048,
                                temperature=0.1,
                            )
                        except Exception:
                            pass
                        finally:
                            _fe.set()
                    threading.Thread(target=_final_call, daemon=True).start()
                    while not _fe.wait(timeout=20):
                        yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                    text = (_fr[0].choices[0].message.content if _fr[0] else None) or "Unable to generate final response."
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
    # Real Estate: units by state (secondary)
    units_by_state = safe("rs.unit", "read_group",
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
        "adv_by_state":             extract(adv_by_state,             "state"),
        "adv_by_project":           extract(adv_by_project,           "project_id", amount_field="amount"),
        "items_by_project":         extract(items_by_project,         "project_id", amount_field="total_cost"),
        "po_by_vendor":             extract(po_by_vendor,             "partner_id",  amount_field="amount_total"),
        "emp_by_dept":              extract(emp_by_dept,              "department_id"),
        "units_by_state":           extract(units_by_state,           "state"),
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
    groq_key = request.form.get("api_key", "").strip() or DEFAULT_GROQ_KEY

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
            client_tmp = Groq(api_key=groq_key)
            resp = client_tmp.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
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
    if not sid or not token:
        logging.warning("Twilio credentials not set — cannot send WhatsApp reply")
        return
    if not _requests:
        logging.warning("requests library not installed")
        return
    auth = base64.b64encode(f"{sid}:{token}".encode()).decode()
    url  = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    try:
        _requests.post(url,
            headers={"Authorization": f"Basic {auth}"},
            data={"From": from_, "To": to_number, "Body": body},
            timeout=10)
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
    groq_key = DEFAULT_GROQ_KEY
    if not groq_key:
        return "❌ Groq API key not configured on the server."

    client  = Groq(api_key=groq_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]

    for _ in range(5):
        try:
            resp = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=2048,
                temperature=0.1,
            )
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
    """Twilio WhatsApp webhook. Returns 200 immediately; replies asynchronously."""
    from_number  = request.form.get("From", "").strip()
    body         = request.form.get("Body", "").strip()
    num_media    = int(request.form.get("NumMedia", "0") or "0")
    media_url    = request.form.get("MediaUrl0", "").strip()
    media_type   = request.form.get("MediaContentType0", "").strip()

    if not from_number:
        return ('<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
                200, {"Content-Type": "text/xml"})

    # ── Voice note ────────────────────────────────────────────────────────────
    if num_media > 0 and media_url and "audio" in media_type:
        def _handle_voice():
            try:
                _twilio_send(from_number, "🎤 جاري تفريغ الصوت وتحليل سؤالك…")
                transcript = _transcribe_voice(media_url)
                if not transcript:
                    _twilio_send(from_number, "❌ لم أتمكن من تفريغ الرسالة الصوتية. حاول مرة أخرى.")
                    return
                # Echo transcript so user knows what was understood
                _twilio_send(from_number, f"🎙️ فهمت: {transcript}")
                # Run AI query on the transcript
                answer = _run_ai_sync(transcript)
                for chunk in _split_whatsapp(_md_to_whatsapp(answer)):
                    _twilio_send(from_number, chunk)
            except Exception as e:
                _twilio_send(from_number, f"❌ خطأ: {str(e)[:200]}")
        threading.Thread(target=_handle_voice, daemon=True).start()

    # ── Text message ──────────────────────────────────────────────────────────
    elif body:
        threading.Thread(
            target=_handle_whatsapp,
            args=(from_number, body),
            daemon=True
        ).start()

    # Must return 200 + empty TwiML quickly (Twilio 15 s timeout)
    return ('<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
            200, {"Content-Type": "text/xml"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
