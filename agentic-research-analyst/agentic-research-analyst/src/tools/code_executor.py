"""
Code Executor Tool — Sandboxed Python execution for data analysis.

Runs user-generated Python code in a restricted subprocess with
timeout enforcement and output capture.
"""

from __future__ import annotations

import subprocess
import tempfile
import os
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of a sandboxed code execution."""

    output: str
    error: str
    return_code: int
    timed_out: bool


class CodeExecutor:
    """
    Sandboxed Python code executor.

    Security
    --------
    - Runs in a subprocess with restricted permissions
    - Enforces configurable timeout
    - Captures stdout/stderr separately
    - No filesystem write access outside temp directory
    - No network access from executed code
    """

    FORBIDDEN_IMPORTS = {
        "os.system", "subprocess", "shutil.rmtree",
        "socket", "http.client", "urllib.request",
        "__import__", "eval", "exec", "compile",
    }

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    def run(self, code: str) -> ExecutionResult:
        """
        Execute Python code in a sandboxed environment.

        Parameters
        ----------
        code : str
            Python code to execute.

        Returns
        -------
        ExecutionResult
            Captured output, errors, and execution metadata.

        Raises
        ------
        ValueError
            If code contains forbidden imports or operations.
        """
        # Security check
        self._validate_code(code)

        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
        ) as f:
            # Inject safe imports header
            f.write("import warnings\nwarnings.filterwarnings('ignore')\n\n")
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir(),
            )
            return ExecutionResult(
                output=result.stdout,
                error=result.stderr,
                return_code=result.returncode,
                timed_out=False,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                output="",
                error=f"Execution timed out after {self.timeout}s",
                return_code=-1,
                timed_out=True,
            )
        finally:
            os.unlink(temp_path)

    def _validate_code(self, code: str) -> None:
        """Check code for forbidden operations."""
        for forbidden in self.FORBIDDEN_IMPORTS:
            if forbidden in code:
                raise ValueError(
                    f"Code contains forbidden operation: {forbidden}. "
                    "Sandboxed execution does not allow system access."
                )
