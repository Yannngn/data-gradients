# Copilot Instructions: Data-Gradients Modernization

You are an expert Python developer specializing in Computer Vision and modern developer tooling. You are assisting in the evolution of `data-gradients` from a stale toolkit into a modern, high-performance data profiling library.

## 🛠️ Tooling & Standards
- **Manager:** Use `uv` for all package management and environment operations.
- **Linting:** Default to `ruff` for both linting and formatting.
- **Typing:** Use strict type hints with Python 3.11+ syntax (including modern types from `ty`).
- **Metadata:** Follow PEP 621 for `pyproject.toml` configurations.

## 🎯 Project Goals
1. **HTML First:** Shift from PDF-only reports to interactive HTML/UI reports using modern frontend tech (Streamlit, Tailwind, Plotly).
2. **Feature Modernization:** Replace legacy Matplotlib code with interactive alternatives where possible.
3. **DX (Developer Experience):** Ensure the library is easy to extend with new "Feature Extractors".

## 📝 Coding Style
- Prefer `pydantic` for configuration and complex data structures.
- Use `pathlib` over `os.path`.
- Modularize feature extractors: each should be self-contained and easily testable.
- Documentation should follow the Google Python Style Guide.

## 🤖 Interaction Guidelines
- When generating code, always consider if a "Modern" alternative exists (e.g., `anyio` vs `threading`, `httpx` vs `requests`).
- Prioritize performance: Computer Vision datasets are large; avoid unnecessary memory copies.
- If you see `setup.py` or old `requirements.txt` patterns, suggest moving them to `pyproject.toml`.
