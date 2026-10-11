# Agentic AI Research Assistant

An AI-powered research assistant built using Python,
Google Agent Development Kit (ADK), Groq and Tavily.

It searches the web, extracts webpage content when needed,
summarizes research findings and saves reports as Markdown files.

## Features

- Web search using Tavily
- Webpage content extraction
- Evidence-based summaries with source URLs
- Markdown research report generation
- Local report storage
- Debug logs for troubleshooting
- Token-conscious search and content limits

## Tech Stack

- Python
- Google ADK
- Groq API
- Qwen model
- Tavily Search and Extract APIs
- python-dotenv

## Project Structure

research agent/
├── requirements.txt
├── .gitignore
└── my_agent/
    ├── agent.py
    ├── .env
    ├── .env.example
    └── reports/

## Setup on Windows

### 1. Open the project directory

Open the project folder in VS Code.

### 2. Activate the virtual environment

In PowerShell, run:

```powershell
.\venv\Scripts\Activate.ps1