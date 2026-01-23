# 🚀 Data-Gradients: The Evolution

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Built with uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v0.json)](https://github.com/astral-sh/ruff)
[![Ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.md)

**Data-Gradients** is being reimagined. Originally an open-source library for computer vision dataset analysis, this fork focuses on **modernizing the internals**, **enhancing the UI**, and **improving the developer experience** using the latest Python ecosystem tools.

---

## ✨ Modernization Goals

This project serves as a demonstration of modernizing "stale" open-source repositories. Key focus areas include:

- [ ] **Modern Tooling:** Migrating from legacy \`setup.py\` to \`uv\` and \`pyproject.toml\` (PEP 621).
- [ ] **Interactive HTML UI:** Replacing static PDF reports with dynamic, interactive HTML dashboards using **Streamlit**.
- [ ] **Performance Boost:** Leveraging \`ruff\` for hyper-fast linting and optimizing data processing pipelines.
- [ ] **Type Safety:** Implementing rigorous type hinting and Pydantic validation across the codebase.
- [ ] **AI-Assisted Development:** Utilizing advanced LLM prompting and automated workflows for code evolution.

## 🛠️ The New Stack

We are building on the shoulders of giants, but with a sharper edge:
- **Orchestration:** [uv](https://github.com/astral-sh/uv) (The fastest Python package manager)
- **Quality:** [Ruff](https://github.com/astral-sh/ruff) (The lightning-fast linter/formatter)
- **Type Checking:** [ty](https://github.com/astral-sh/ty) (Modern type checking for humans)
- **UI:** [Streamlit](https://streamlit.io/) + Plotly (Interactive data exploration)
- **Core:** PyTorch & Computer Vision primitives

## 🚀 Getting Started (Experimental)

Ensure you have \`uv\` installed.

\`\`\`bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/data-gradients-evolution.git
cd data-gradients-evolution

# Sync the environment
uv sync

# Run the legacy analysis (Modern UI coming soon)
uv run python examples/segmentation_example.py
\`\`\`

## 📈 Roadmap

1.  **Phase 1 (Foundation):** Convert project to \`uv\` based structure, clean up legacy dependencies.
2.  **Phase 2 (HTML Reporting):** Implement a new \`HTMLReportManager\` using Streamlit and Plotly.
3.  **Phase 3 (Interactivity):** Add Plotly charts for deep-dive dataset exploration.
4.  **Phase 4 (API Refactor):** Simplify the "Extractor" registration system.

## 🤝 Contribution & Feedback

This is an active portfolio evolution. If you have ideas on modernizing CV data tools, feel free to open an issue or reach out.

---

*Original library by Deci AI. Modernized with ❤️ by @Yannngn*
