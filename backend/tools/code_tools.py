import ast
import subprocess
import sys


def is_code_safe(code):
    if not code:
        return False, "Empty code."
    lowered = code.lower()
    deny = [
        "import os",
        "import sys",
        "import subprocess",
        "import socket",
        "import shutil",
        "import pathlib",
        "open(",
        "exec(",
        "eval(",
        "__import__",
        "pickle",
    ]
    for token in deny:
        if token in lowered:
            return False, f"Blocked token: {token}"
    if len(code) > 2000:
        return False, "Code too long."
    return True, ""


def validate_python_syntax(code):
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})"


def execute_python(code, timeout=4):
    safe, reason = is_code_safe(code)
    if not safe:
        return {"stdout": "", "stderr": f"Execution blocked: {reason}", "returncode": 1}
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Execution timed out.", "returncode": 1}
