
# Football Odds Intelligence AI

## Overview

A machine learning project that estimates football
match outcome probabilities for the 1X2 market:
home win, draw, and away win.

## Long-Term Goal

Build a football probability prediction system that supports
multiple betting markets. The current initial scope is 1X2
probability prediction; support for additional markets will come
in later development phases.

## Development Phases

1. Predict 1X2 probabilities (Home Win, Draw, Away Win).
2. Develop goal prediction models using home and away goals.
3. Calculate Asian Handicap probabilities.
4. Add Over/Under and Both Teams To Score (BTTS) probabilities.

Complete and properly evaluate the 1X2 baseline before
implementing Handicap or any other additional markets.

The project is educational and research-oriented.
It does not guarantee betting profits.

## Architecture Principles

- Keep prediction models separate from betting market calculations.
- Design future components to support multiple betting markets.
- Preserve home goals and away goals data for future goal prediction.
- Use goal prediction models to support market calculations in later phases.

## Project Activities

- Explore historical football match and odds data.
- Build probability estimation models.
- Compare model predictions with market-implied probabilities.
- Evaluate predictions using proper scoring rules.
- Present results through a simple dashboard.

## Initial Dataset

- Competition: English Premier League
- Season: 2025/26
- Source: https://www.football-data.co.uk/

## Tech Stack

- Python
- pandas and NumPy
- scikit-learn
- FastAPI
- Streamlit
- pytest

## Current Progress

Day 1: Project setup and initial dataset inspection.

## Disclaimer

This project is for educational and statistical research.
Predicted probabilities are uncertain and do not guarantee
profitable betting outcomes.
