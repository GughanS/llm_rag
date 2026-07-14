"""
Phase 0 verification script.
Checks that all expected directories, files, and ADR content exist.
"""
import os
import sys

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

# --- 1. Directory structure check ---
EXPECTED_DIRS = [
    "docs",
    "docs/adr",
    "model",
    "kernels",
    "distributed",
    "alignment",
    "serving",
    "tests",
    "tests/unit",
    "tests/integration",
    "tests/e2e",
    ".github/workflows",
    "monitoring",
    "infra",
]

EXPECTED_FILES = [
    "README.md",
    ".gitignore",
    "requirements.txt",
    "docs/PR_TEMPLATE.md",
    "docs/refactor-notes.md",
    "docs/adr/0001-architecture.md",
    "model/__init__.py",
    "kernels/__init__.py",
    "distributed/__init__.py",
    "alignment/__init__.py",
    "serving/__init__.py",
    "tests/__init__.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    "tests/e2e/__init__.py",
]

# --- 2. ADR content checks ---
ADR_REQUIRED_SECTIONS = [
    "Testable Requirements",
    "Monolith",
    "SQLite",
    "Redis",
    "Rate Limiting",
    "Message Brokers",
    "Kafka",
    "GHCR",
    "Prometheus",
    "Grafana",
    "PagerDuty",
    "ELK",
    "Security",
    "safetensors",
    "pip-audit",
    "OAuth2",
    "Scale-trigger",
    "[BUILD]",
    "[DOCUMENT ONLY]",
]


def main():
    errors = []
    passes = []

    # Check directories
    for d in EXPECTED_DIRS:
        full_path = os.path.join(WORKSPACE, d)
        if os.path.isdir(full_path):
            passes.append(f"DIR  OK   {d}/")
        else:
            errors.append(f"DIR  FAIL {d}/ — not found")

    # Check files
    for f in EXPECTED_FILES:
        full_path = os.path.join(WORKSPACE, f)
        if os.path.isfile(full_path):
            size = os.path.getsize(full_path)
            passes.append(f"FILE OK   {f} ({size} bytes)")
        else:
            errors.append(f"FILE FAIL {f} — not found")

    # Check ADR content
    adr_path = os.path.join(WORKSPACE, "docs/adr/0001-architecture.md")
    if os.path.isfile(adr_path):
        with open(adr_path, "r", encoding="utf-8") as fh:
            adr_content = fh.read()
        for keyword in ADR_REQUIRED_SECTIONS:
            if keyword.lower() in adr_content.lower():
                passes.append(f"ADR  OK   contains '{keyword}'")
            else:
                errors.append(f"ADR  FAIL missing '{keyword}'")
    else:
        errors.append("ADR  FAIL 0001-architecture.md not found — skipping content checks")

    # Report
    print("=" * 60)
    print("Phase 0 Verification Report")
    print("=" * 60)
    for p in passes:
        print(f"  [PASS] {p}")
    if errors:
        print()
        for e in errors:
            print(f"  [FAIL] {e}")
    print()
    print(f"PASSED: {len(passes)}  |  FAILED: {len(errors)}")
    print("=" * 60)

    if errors:
        sys.exit(1)
    else:
        print("\nPhase 0 verification PASSED -- all checks green.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
