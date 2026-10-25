# Project Instructions

## Project goal

Build an educational football odds analysis project that estimates calibrated probabilities for a match ending in a home win, draw, or away win (1X2), and compares those estimates with bookmaker-implied probabilities.

Predictions are uncertain. Never claim that the model can guarantee profitable betting outcomes.

## Explain the work clearly

The project owner is a beginner in machine learning. Explain important concepts, formulas, design choices, and trade-offs in beginner-friendly language. Define unfamiliar terms when they first appear.

Work in small steps. Propose or implement no more than five implementation steps at a time.

Explain how to run scripts from the Terminal. When adding or changing tests, explain how to run them.

## Python environment and dependencies

Use the existing Conda environment named `football-odds`.

Do not create another environment or install, upgrade, or remove packages without the owner's explicit approval.

## Data handling and leakage prevention

Keep raw datasets out of Git. Do not print or commit a full dataset. Show only small, relevant samples or summaries when inspecting data.

Prevent data leakage: for pre-match predictions, use only information that would have been available before kickoff. Do not use match results or other information recorded during or after the match as prediction inputs.

Use time-based validation when evaluating models. Do not use random train/test splitting for model evaluation, because it can allow future matches to influence evaluation of earlier matches.

## Code quality and tests

Write readable, modular Python code. Use type hints where they improve clarity, and add docstrings to non-obvious functions.

Add tests for important data transformations and model behavior. Keep tests understandable and explain what they check.

## Approval and change management

Do not modify files, install packages, download data, or run destructive commands without the owner's explicit approval.

Before each task that may change the project or its environment, state the intended changes and list the files or resources that would be affected. Wait for approval before carrying out those changes.

After approved implementation, summarize:
- What changed and which files were affected
- Tests or checks run and their results
- Any remaining issues or limitations
