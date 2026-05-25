"""
Unit tests for individual tools.
"""

import pytest
from src.tools.code_executor import CodeExecutor
from src.tools.citation_linker import CitationLinker


class TestCodeExecutor:
    def setup_method(self):
        self.executor = CodeExecutor(timeout=10)

    def test_simple_execution(self):
        result = self.executor.run("print('hello world')")
        assert result.output.strip() == "hello world"
        assert result.return_code == 0
        assert not result.timed_out

    def test_math_execution(self):
        result = self.executor.run("print(2 + 2)")
        assert result.output.strip() == "4"

    def test_timeout_enforcement(self):
        executor = CodeExecutor(timeout=1)
        result = executor.run("import time; time.sleep(5)")
        assert result.timed_out

    def test_forbidden_imports_blocked(self):
        with pytest.raises(ValueError, match="forbidden operation"):
            self.executor.run("import subprocess; subprocess.run(['ls'])")

    def test_forbidden_os_system(self):
        with pytest.raises(ValueError, match="forbidden operation"):
            self.executor.run("import os; os.system('ls')")

    def test_multiline_output(self):
        code = "for i in range(3): print(f'line {i}')"
        result = self.executor.run(code)
        assert "line 0" in result.output
        assert "line 2" in result.output


class TestCitationLinker:
    def setup_method(self):
        self.linker = CitationLinker()

    def test_link_basic(self):
        sections = [
            {"heading": "Overview", "content": "The market grew by 40% [1]. Revenue reached $5B [2]."}
        ]
        evidence = [
            {"url": "https://example.com/1", "title": "Report A", "content": "Market grew by 40%", "similarity": 0.9},
            {"url": "https://example.com/2", "title": "Report B", "content": "Revenue reached $5B", "similarity": 0.85},
        ]

        citations = self.linker.link(sections, evidence)
        assert len(citations) == 2
        assert citations[0].source_url == "https://example.com/1"
        assert citations[1].source_url == "https://example.com/2"

    def test_link_no_markers(self):
        sections = [{"heading": "Overview", "content": "No citations here."}]
        citations = self.linker.link(sections, [])
        assert len(citations) == 0

    def test_build_graph(self):
        sections = [{"heading": "Test", "content": "Claim [1]."}]
        evidence = [{"url": "https://src.com", "title": "Source", "content": "data", "similarity": 0.8}]

        citations = self.linker.link(sections, evidence)
        graph = self.linker.build_graph(citations)
        assert "https://src.com" in graph["sources"]
        assert len(graph["claims"]) == 1
