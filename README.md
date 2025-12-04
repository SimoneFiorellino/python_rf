# Machine Learning Models – Setup Guide

This project uses:

- [uv](https://github.com/astral-sh/uv) for environment & dependency management  
- Ruff + pre-commit for automatic formatting and linting  

Below are the **minimal terminal commands** needed to fully set up and replicate the project.

# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo
git clone <YOUR_REPO_URL>.git
cd <YOUR_REPO_FOLDER>

# Sync environment and dependencies
uv sync

# (Optional) Refresh environment if project name changed
uv venv --refresh
uv sync

# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit once over all files
uv run pre-commit run --all-files

# Run unit tests
uv run pytest

# Run your Python scripts
uv run -m src.main

# Commit workflow
git add .
git commit -m "message"

