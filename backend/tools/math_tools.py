import re
import sympy as sp
from sympy.printing.mathml import mathml


def _normalize_expr(expr):
    expr = expr.replace("^", "**")
    expr = expr.replace(",", "")
    return expr.strip()


def compute_expression(expr):
    if not expr:
        return None
    expr = _normalize_expr(expr)
    try:
        sym = sp.sympify(expr)
        simplified = sp.simplify(sym)
        if simplified.is_number:
            value = float(simplified)
            result = str(value).rstrip("0").rstrip(".") if "." in str(value) else str(value)
        else:
            result = str(simplified)
        latex = sp.latex(simplified)
        mathml_out = mathml(simplified)
        return {"result": result, "latex": latex, "mathml": mathml_out}
    except Exception:
        return None


def solve_equation(equation, symbol=None):
    if not equation:
        return None
    equation = _normalize_expr(equation)
    try:
        if "=" in equation:
            left, right = equation.split("=", 1)
            left_expr = sp.sympify(left)
            right_expr = sp.sympify(right)
            eq = sp.Eq(left_expr, right_expr)
        else:
            eq = sp.Eq(sp.sympify(equation), 0)

        if symbol:
            sym = sp.symbols(symbol)
        else:
            free = list(eq.free_symbols)
            sym = free[0] if free else sp.symbols("x")

        solutions = sp.solve(eq, sym)
        sol_text = [str(s) for s in solutions] if solutions else []
        sol_latex = [sp.latex(s) for s in solutions] if solutions else []
        sol_mathml = [mathml(s) for s in solutions] if solutions else []
        return {"symbol": str(sym), "solutions": sol_text, "latex": sol_latex, "mathml": sol_mathml}
    except Exception:
        return None


def extract_equation(text):
    if not text:
        return ""
    match = re.search(r"([0-9a-zA-Z_+*/^().\s-]+=[0-9a-zA-Z_+*/^().\s-]+)", text)
    if match:
        eq = match.group(1).strip()
        eq = re.sub(r"^(solve|equation|solve for\s+[a-z])\s+", "", eq, flags=re.IGNORECASE)
        return eq.strip()
    return ""


def extract_expression(text):
    if not text:
        return ""
    match = re.search(r"([-+*/^().0-9\s]+)", text)
    if match:
        return match.group(1).strip()
    return ""


def render_math(expr):
    if not expr:
        return None
    expr = _normalize_expr(expr)
    try:
        sym = sp.sympify(expr)
        return {"latex": sp.latex(sym), "mathml": mathml(sym)}
    except Exception:
        return None
