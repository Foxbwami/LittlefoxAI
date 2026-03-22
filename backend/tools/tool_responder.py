import re
from backend.core import config
from backend.tools.math_tools import compute_expression, solve_equation, extract_equation, extract_expression, render_math
from backend.tools.code_tools import validate_python_syntax, execute_python


def handle_tool_request(user_input, allow_execute=False):
    text = user_input or ""
    lower = text.lower()
    code = _extract_code(text)

    if code and ("validate syntax" in lower or "check syntax" in lower):
        ok, error = validate_python_syntax(code)
        return "Syntax OK." if ok else error

    if code and any(k in lower for k in ["run", "execute", "run this code", "execute this code"]):
        if not (allow_execute and config.ALLOW_CODE_EXECUTION):
            return "Execution is disabled. Send `execute: true` and enable `ALLOW_CODE_EXECUTION` in config."
        result = execute_python(code, timeout=config.CODE_EXEC_TIMEOUT)
        output = result.get("stdout") or ""
        err = result.get("stderr") or ""
        if err:
            return f"Execution error:\n{err}"
        if output:
            return f"Output:\n{output}"
        return "Execution finished with no output."

    if any(k in lower for k in ["solve", "equation", "solve for"]):
        eq = extract_equation(text)
        if eq:
            solved = solve_equation(eq)
            if solved:
                latex = ""
                if config.MATH_RENDER and solved.get("latex"):
                    latex = "LaTeX: $$" + ", ".join(solved["latex"]) + "$$"
                sol_text = ", ".join(solved["solutions"]) if solved["solutions"] else "No solutions found."
                return f"Solution for {solved['symbol']}: {sol_text}\n{latex}".strip()

    if any(k in lower for k in ["calculate", "compute", "evaluate", "result of"]):
        expr = extract_expression(text)
        if expr:
            computed = compute_expression(expr)
            if computed:
                latex = ""
                if config.MATH_RENDER and computed.get("latex"):
                    latex = f"LaTeX: $$ {computed['latex']} $$"
                return f"Result: {computed['result']}\n{latex}".strip()

    if _looks_like_expression(text):
        computed = compute_expression(text)
        if computed:
            latex = ""
            if config.MATH_RENDER and computed.get("latex"):
                latex = f"LaTeX: $$ {computed['latex']} $$"
            return f"Result: {computed['result']}\n{latex}".strip()

    return None


def handle_tool_request_structured(tool, payload, allow_execute=False):
    tool = (tool or "").lower()
    if tool == "compute":
        expr = payload.get("expression", "")
        return compute_expression(expr) or {}
    if tool == "solve":
        eq = payload.get("equation", "")
        symbol = payload.get("symbol")
        return solve_equation(eq, symbol=symbol) or {}
    if tool == "validate_syntax":
        code = payload.get("code", "")
        ok, error = validate_python_syntax(code)
        return {"ok": ok, "error": error}
    if tool == "execute":
        code = payload.get("code", "")
        if not (allow_execute and config.ALLOW_CODE_EXECUTION):
            return {"stdout": "", "stderr": "Execution disabled.", "returncode": 1}
        return execute_python(code, timeout=config.CODE_EXEC_TIMEOUT)
    if tool == "render_math":
        expr = payload.get("expression", "")
        return render_math(expr) or {}
    return {"error": "Unknown tool."}


def _extract_code(text):
    if not text:
        return ""
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"`(?:python)?\n?(.*?)`", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    if text.lower().startswith("/run "):
        return text.split(" ", 1)[1].strip()
    if "code:" in text.lower():
        parts = text.split("code:", 1)
        return parts[1].strip()
    lowered = text.lower()
    for prefix in ["validate syntax", "check syntax", "run", "execute"]:
        if lowered.startswith(prefix):
            remainder = text[len(prefix):].lstrip(" :-\n")
            return remainder.strip()
    match = re.search(r"(?:validate syntax|check syntax|run|execute)\s*[:\-]?\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _looks_like_expression(text):
    return bool(re.match(r"^[0-9\s\\+\\-\\*/\\^\\(\\)\\.]+$", text.strip()))
