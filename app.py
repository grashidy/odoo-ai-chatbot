import xmlrpc.client, json, os, time, re
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from groq import Groq

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
                    "model":  {"type": "string", "description": "Odoo model. E.g. rs.unit, rs.project, hr.employee, rs.contract, rs.installment, purchase.order, construction.advance.payment"},
                    "domain": {"type": "string", "description": "JSON filter string. '[]'=all. E.g. '[[\"state\",\"=\",\"sale\"]]'"},
                    "fields": {"type": "string", "description": "JSON array of field names. E.g. '[\"name\",\"state\",\"rs_project_id\"]'"},
                    "limit":  {"type": "integer", "description": "Max rows (default 50, max 200)"},
                    "order":  {"type": "string",  "description": "Sort. E.g. 'current_sale_price desc'"}
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
                    "domain": {"type": "string", "description": "JSON filter. '[]'=all."}
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
            "description": "Group & count Odoo records. Use for 'how many X per Y' statistics and charts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Odoo model name."},
                    "domain": {"type": "string", "description": "Filter domain as JSON string. Use '[]' for all. Example: '[[\"state\",\"=\",\"draft\"]]'"},
                    "groupby": {
                        "type": "string",
                        "description": "Fields to group by as JSON string. E.g. '[\"rs_project_id\"]' or '[\"state\"]' or '[\"department_id\"]'."
                    },
                    "aggregates": {
                        "type": "string",
                        "description": "Optional: numeric fields to sum as JSON string. E.g. '[\"net_area:sum\",\"current_sale_price:sum\"]'. Use '[]' if none."
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
        # bare single field name like "rs_project_id"
        return [raw] if raw else []
    return []

def run_tool(name, args):
    try:
        if name == "odoo_search":
            domain = _coerce_domain(args.get("domain", []))
            limit  = min(int(args.get("limit", 50)), 200)
            fields = _coerce_list(args.get("fields", ["display_name"]))
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
            domain  = _coerce_domain(args.get("domain", []))
            groupby = _coerce_list(args.get("groupby", []))
            agg_fields  = _coerce_list(args.get("aggregates", []))
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
SYSTEM_PROMPT = """You are an AI assistant for a real estate & construction company on Odoo 18.
Reply in same language as user (Arabic or English).

RULES:
- Use tools for real data. Never guess numbers.
- Each tool call at most ONCE. No repeating same tool.
- Counts/stats: odoo_read_group. Records: odoo_search (limit 50).
- Format numbers with commas. Prices in EGP. Use markdown tables.
- On field error: call odoo_get_fields once to check fields, then retry.

CHARTS (include when showing statistics):
CHART_BAR:{"title":"T","labels":["A","B"],"data":[10,20]}
CHART_PIE:{"title":"T","labels":["A","B"],"data":[10,20]}

MODELS:
Real Estate: rs.project, rs.unit(unit_code,state,rs_project_id,net_area,current_sale_price,partner_id), rs.contract(partner_id,rs_unit_id,state,contracted_sale_price), rs.installment(partner_id,amount,date,state), rs.rsrvrq, rs.eoi
Construction BOQ: boq.contract(partner_id,project_id), project.subcontracting.boq.line(name,boq_contract_id,project_id,product_id,quantity,billed_qty,remain_qty,boq_cost,work_type), project.detailed.item.line(name,project_id,quantity,done_qty,initial_cost,actual_cost,total_cost,progress_percentage)
Payments: construction.advance.payment(name,partner_id,amount,date,state,project_id,due_amount,settled_amount,subcontractor_contract_id)
Purchases: purchase.order(name,partner_id,state,amount_total,date_order,project_id), purchase.order.line(order_id,product_id,product_qty,price_unit,price_subtotal)
Tasks: project.task(name,project_id,stage_id,date_deadline,kanban_state,user_ids), project.project(name,user_id)
HR: hr.employee(name,department_id,job_title,work_phone,mobile_phone), hr.department(name,manager_id)
Other: res.partner(name,phone,mobile,email)

BOQ notes: quantity=planned, billed_qty=actual done. Over budget = billed_qty > quantity.
FIELD-TO-FIELD COMPARISONS: Odoo domain CANNOT compare two fields (e.g. billed_qty > quantity is invalid in domain). For these queries, call odoo_search with domain=[] and fields=["name","quantity","billed_qty","remain_qty","project_id"] limit=200, then in your answer list only the rows where billed_qty > quantity based on the data returned. Never put a field name as the value in a domain filter.
Use odoo_search not odoo_read_group for field comparisons."""

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

            max_iterations = 5
            tool_call_counts = {}   # tool_name → how many times called total

            for iteration in range(max_iterations):
                response = None
                try:
                    response = client.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        max_tokens=4096,
                        temperature=0.1,
                    )
                except Exception as api_err:
                    err_msg = str(api_err)
                    is_rate = "rate_limit" in err_msg.lower() or "429" in err_msg
                    if is_rate:
                        # Parse actual wait time from Groq error message
                        m_wait = re.search(r'try again in ([\d.]+)s', err_msg, re.IGNORECASE)
                        wait_sec = int(float(m_wait.group(1))) + 2 if m_wait else 65
                        if "per day" in err_msg.lower() or "tpd" in err_msg.lower() or wait_sec > 300:
                            yield f"data: {json.dumps({'type': 'text', 'text': '⚠️ Daily API quota exhausted. Please wait a few hours or use a new Groq API key (free at console.groq.com).'})}\n\n"
                            break
                        else:
                            # Auto-wait and retry — don't make user do it manually
                            yield f"data: {json.dumps({'type': 'tool', 'name': 'wait', 'input': {'model': f'Waiting {wait_sec}s for rate limit to reset...'}})}\n\n"
                            time.sleep(wait_sec)
                            continue  # retry this iteration
                    else:
                        yield f"data: {json.dumps({'type': 'text', 'text': f'❌ API Error: {err_msg[:250]}'})}\n\n"
                    break

                if response is None:
                    break

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
                    # Force one final answer call with no more tools
                    messages.append({
                        "role": "user",
                        "content": "You have enough data. Stop calling tools and give the final answer now."
                    })
                    try:
                        final = client.chat.completions.create(
                            model="meta-llama/llama-4-scout-17b-16e-instruct",
                            messages=messages,
                            max_tokens=2048,
                            temperature=0.1,
                        )
                        text = final.choices[0].message.content or "No response."
                    except Exception:
                        text = "Unable to generate final response."
                    yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"
                    break

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

    # Real Estate: units by state
    units_by_state = safe("rs.unit", "read_group", [[], ["state"], ["state"]], {"lazy": False})
    # Real Estate: units by project
    units_by_project = safe("rs.unit", "read_group", [[], ["rs_project_id"], ["rs_project_id"]], {"lazy": False})
    # Real Estate: installments by state
    install_by_state = safe("rs.installment", "read_group", [[], ["state"], ["state"]], {"lazy": False})
    # HR: employees by department
    emp_by_dept = safe("hr.employee", "read_group", [[], ["department_id"], ["department_id"]], {"lazy": False})
    # Purchases: top 10 vendors by total
    po_by_vendor = safe("purchase.order", "read_group",
        [[["state", "in", ["purchase", "done"]]], ["partner_id", "amount_total:sum"], ["partner_id"]],
        {"lazy": False, "limit": 10, "orderby": "amount_total desc"})
    # Advance payments summary
    adv_by_state = safe("construction.advance.payment", "read_group", [[], ["state"], ["state"]], {"lazy": False})
    # Advance payments by project
    adv_by_project = safe("construction.advance.payment", "read_group",
        [[], ["project_id", "amount:sum"], ["project_id"]], {"lazy": False})
    # BOQ contracts count by project
    boq_by_project = safe("boq.contract", "read_group", [[], ["project_id"], ["project_id"]], {"lazy": False})

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
        "units_by_state":    extract(units_by_state,    "state"),
        "units_by_project":  extract(units_by_project,  "rs_project_id"),
        "install_by_state":  extract(install_by_state,  "state"),
        "emp_by_dept":       extract(emp_by_dept,       "department_id"),
        "po_by_vendor":      extract(po_by_vendor,      "partner_id", amount_field="amount_total"),
        "adv_by_state":      extract(adv_by_state,      "state"),
        "adv_by_project":    extract(adv_by_project,    "project_id", amount_field="amount"),
        "boq_by_project":    extract(boq_by_project,    "project_id"),
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
