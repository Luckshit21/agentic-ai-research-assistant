from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
import ipaddress
import os
import re
import time

from dotenv import load_dotenv
from tavily import TavilyClient
from google.genai import types
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"

load_dotenv(BASE_DIR / ".env")

DEBUG = os.getenv("AGENT_DEBUG", "1").lower() not in {
    "0", "false", "no"
}

MAX_RESULTS_PER_SEARCH = 2
MAX_RESULT_CHARS = 250
MAX_EXTRACT_CHARS = 1200
MAX_REPORT_CHARS = 15000

RESEARCH_TERMS = (
    "research",
    "report",
    "analysis",
    "analyze",
    "analyse",
    "compare",
    "comparison",
    "investigate",
    "investigation",
    "detailed",
    "deep dive",
    "comprehensive",
    "evaluate",
    "evaluation",
    "study",
    "assessment",
    "review",
    "research paper",
)

NEWS_TERMS = (
    "latest",
    "news",
    "today",
    "recent",
    "current",
    "breaking",
)


def debug(component, message):
    """Print concise diagnostic logs without exposing secrets."""
    if DEBUG:
        print(f"[DEBUG:{component}] {message}", flush=True)


def clean_url(url):
    return (url or "").strip().rstrip(".,;:)]}")


def get_tavily_client():
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is missing from my_agent/.env"
        )

    return TavilyClient(api_key=api_key)


def is_research_request(text):
    """Detect research/report requests across any subject."""
    text = (text or "").lower()

    return any(term in text for term in RESEARCH_TERMS)


def is_news_request(text):
    text = (text or "").lower()
    return any(term in text for term in NEWS_TERMS)


def extract_title(text):
    """Read an optional title from: title 'My Report'."""
    match = re.search(
        r"""title\s+["']([^"']+)["']""",
        text,
        flags=re.IGNORECASE,
    )

    return match.group(1).strip() if match else "Research Report"


def make_search_query(text):
    """Convert the request into a compact, topic-neutral search query."""

    lines = [
        line.strip()
        for line in (text or "").splitlines()
        if line.strip()
    ]

    query = lines[0] if lines else (text or "")

    # Avoid searching for report-formatting instructions.
    query = re.sub(
        r"""title\s+["'][^"']+["']""",
        "",
        query,
        flags=re.IGNORECASE,
    )

    query = re.sub(
        r"^(please\s+)?(?:do\s+)?(?:a\s+)?"
        r"(research|search|find|investigate|analyze|analyse)"
        r"\s+(on\s+|about\s+|for\s+)?",
        "",
        query,
        flags=re.IGNORECASE,
    )

    query = " ".join(query.split()).strip()

    return query[:250] or (text or "")[:250]


# ==========================================================
# 2. WEB SEARCH
# ==========================================================

def run_search(client, query, topic="general"):
    """Run one Tavily search and return raw result records."""

    params = {
        "query": query[:250],
        "max_results": MAX_RESULTS_PER_SEARCH,
        "search_depth": "basic",
        "topic": topic,
    }

    if topic == "news":
        params["time_range"] = "month"

    response = client.search(**params)

    return response.get("results", [])


def format_results(results):
    """Turn raw results into compact evidence for the model."""

    sections = []

    for item in results:
        url = clean_url(item.get("url", ""))

        sections.append(
            f"Title: {item.get('title') or 'N/A'}\n"
            f"URL: {url or 'N/A'}\n"
            f"Date: {item.get('published_date') or 'Unknown'}\n"
            f"Content: "
            f"{(item.get('content') or '')[:MAX_RESULT_CHARS]}"
        )

    return "\n\n---\n\n".join(sections)


def web_search(query: str) -> str:
    """Public helper: perform one compact web search."""

    query = (query or "").strip()[:250]

    if not query:
        return "Search failed: empty query."

    start = time.perf_counter()
    debug("SEARCH", f"START query={query!r}")

    try:
        topic = "news" if is_news_request(query) else "general"
        results = run_search(get_tavily_client(), query, topic)

        debug(
            "SEARCH",
            f"OK results={len(results)} "
            f"seconds={time.perf_counter() - start:.2f}",
        )

        if not results:
            return "No relevant search results were found."

        return format_results(results)

    except Exception as exc:
        debug(
            "SEARCH_ERROR",
            f"{type(exc).__name__}: {str(exc)[:180]}",
        )
        return "Search failed. Do not invent search results."


def search_research_sources(query, detailed):
    """
    Generic research retrieval.

    Normal factual request: one search.
    Research/report request: up to two targeted searches.
    No topic-specific branches for finance, games, health, etc.
    """

    client = get_tavily_client()

    primary_topic = (
        "news" if is_news_request(query) else "general"
    )

    all_results = []
    seen_urls = set()

    queries = [(query, primary_topic)]

    if detailed:
        secondary_query = (
            f"{query} official sources primary research "
            "reports evidence"
        )[:250]

        queries.append((secondary_query, "general"))

    for number, (search_query, topic) in enumerate(queries, start=1):
        started = time.perf_counter()

        try:
            results = run_search(client, search_query, topic)
            added = 0

            for item in results:
                url = clean_url(item.get("url", ""))

                if not url or url in seen_urls:
                    continue

                seen_urls.add(url)
                all_results.append(item)
                added += 1

            debug(
                "SEARCH",
                f"Query {number}/{len(queries)} "
                f"results={len(results)} new={added} "
                f"seconds={time.perf_counter() - started:.2f}",
            )

        except Exception as exc:
            debug(
                "SEARCH_ERROR",
                f"Query {number}: {type(exc).__name__}: "
                f"{str(exc)[:150]}",
            )

    return all_results


def get_search_urls(results):
    """Return unique source URLs from Tavily search records."""

    urls = []

    for item in results:
        url = clean_url(item.get("url", ""))

        if url and url not in urls:
            urls.append(url)

    return urls


# ==========================================================
# 3. WEBPAGE EXTRACTION
# ==========================================================

def validate_public_url(url):
    """Basic URL checks before sending a URL to Tavily."""

    url = (url or "").strip()
    parsed = urlparse(url)

    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        return False

    host = parsed.hostname.lower()

    if host == "localhost" or host.endswith(
        (".localhost", ".local", ".internal")
    ):
        return False

    try:
        address = ipaddress.ip_address(host)

        if not address.is_global:
            return False

    except ValueError:
        pass

    return True


def extract_webpage(url: str, focus: str = "") -> str:
    """Extract readable content from one public webpage."""

    url = clean_url(url)

    if not validate_public_url(url):
        return "Invalid or unsupported public webpage URL."

    started = time.perf_counter()
    debug("EXTRACT", f"START url={url[:150]}")

    try:
        params = {
            "urls": [url],
            "extract_depth": "basic",
        }

        if focus.strip():
            params["query"] = focus.strip()[:150]
            params["chunks_per_source"] = 1

        response = get_tavily_client().extract(**params)
        results = response.get("results", [])

        if not results:
            debug("EXTRACT", "NO_RESULTS")
            return "No readable webpage content could be extracted."

        item = results[0]
        content = item.get("raw_content") or ""
        actual_url = clean_url(item.get("url") or url)

        debug(
            "EXTRACT",
            f"OK chars={len(content)} "
            f"seconds={time.perf_counter() - started:.2f}",
        )

        return (
            f"Source URL: {actual_url}\n"
            f"Extracted content:\n"
            f"{content[:MAX_EXTRACT_CHARS]}"
        )

    except Exception as exc:
        debug(
            "EXTRACT_ERROR",
            f"{type(exc).__name__}: {str(exc)[:180]}",
        )
        return "Webpage extraction failed. Do not invent its contents."


def choose_source_for_extraction(results):
    """
    Prefer apparently authoritative sources when available.
    This is a selection heuristic, not proof of trustworthiness.
    """

    def score(item):
        url = clean_url(item.get("url", ""))
        host = urlparse(url).hostname or ""
        host = host.lower()
        path = url.lower()
        value = 0

        if host.endswith((".gov", ".gov.in", ".edu", ".ac.in")):
            value += 4

        if any(domain in host for domain in (
            "who.int", "un.org", "europa.eu", " nasa.gov"
        )):
            value += 4

        if any(word in path for word in (
            "official", "publication", "research",
            "investor", "press-release", "annual-report",
        )):
            value += 2

        return value

    if not results:
        return ""

    ordered = sorted(results, key=score, reverse=True)

    return clean_url(ordered[0].get("url", ""))


# ==========================================================
# 4. REPORT SAVER
# ==========================================================

def save_research_report(title: str, report_markdown: str) -> str:
    """Save a Markdown report to my_agent/reports."""

    title = (title or "").strip()[:100]
    report_markdown = (report_markdown or "").strip()

    if not title or not report_markdown:
        return "Save failed: title or report is empty."

    if len(report_markdown) > MAX_REPORT_CHARS:
        return "Save failed: report exceeds the character limit."

    slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        title.lower(),
    ).strip("-")

    slug = (slug or "research-report")[:50]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{slug}.md"

    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / filename

        document = (
            f"# {title}\n\n"
            f"Saved: "
            f"{datetime.now().astimezone().isoformat(timespec='minutes')}"
            f"\n\n"
            f"{report_markdown}\n"
        )

        path.write_text(document, encoding="utf-8")

        debug(
            "SAVE",
            f"OK file={filename} chars={len(document)}",
        )

        return f"reports/{filename}"

    except Exception as exc:
        debug(
            "SAVE_ERROR",
            f"{type(exc).__name__}: {str(exc)[:180]}",
        )
        return "Report saving failed."


# ==========================================================
# 5. SOURCE PRESERVATION
# ==========================================================

def add_missing_sources(report, source_urls):
    """Ensure URLs retrieved by the pipeline remain in the response."""

    existing = {
        clean_url(url)
        for url in re.findall(
            r"https?://[^\s<>\])}]+",
            report,
        )
    }

    missing = [
        clean_url(url)
        for url in source_urls
        if clean_url(url) and clean_url(url) not in existing
    ]

    if not missing:
        debug("SOURCES", "All retrieved URLs already appear")
        return report

    has_heading = bool(
        re.search(r"(?im)^#{1,3}\s+sources\b", report)
    )

    heading = (
        "### Additional retrieved sources"
        if has_heading
        else "## Sources"
    )

    report = (
        report.rstrip()
        + "\n\n"
        + heading
        + "\n"
        + "\n".join(f"- {url}" for url in missing)
    )

    debug("SOURCES", f"Added {len(missing)} missing URLs")

    return report


def is_incomplete_answer(answer):
    """Avoid saving obvious refusals as completed research reports."""

    text = (answer or "").lower()

    refusal_markers = (
        "i cannot complete the requested research report",
        "i can't complete the requested research report",
        "unable to complete this report",
        "no usable search results were found",
        "search failed. do not invent",
    )

    return any(marker in text for marker in refusal_markers)


# ==========================================================
# 6. CONTROLLED, TOPIC-INDEPENDENT PIPELINE
# ==========================================================

def before_model_callback(callback_context, llm_request):
    """
    Run retrieval in Python, then send compact evidence to the model.
    Keep only the latest user turn to reduce historical context.
    """

    user_index = None
    user_text = ""

    for index in range(len(llm_request.contents) - 1, -1, -1):
        item = llm_request.contents[index]

        if item.role == "user":
            user_index = index

            user_text = "\n".join(
                part.text
                for part in (item.parts or [])
                if getattr(part, "text", None)
            )
            break

    if user_index is None or not user_text.strip():
        debug("PIPELINE", "No user message found")
        return None

    lower = user_text.lower()

    research_task = is_research_request(user_text)

    explicit_save = bool(
        re.search(r"\b(save|export)\b", lower)
    )

    # Automatically save research/report tasks.
    # Simple factual questions are not automatically saved.
    save_requested = explicit_save or research_task

    query = make_search_query(user_text)
    report_title = extract_title(user_text)

    callback_context.state["temp:save_report"] = save_requested
    callback_context.state["temp:report_title"] = report_title
    callback_context.state["temp:research_task"] = research_task

    debug(
        "PIPELINE",
        f"START research={research_task} save={save_requested}",
    )

    # Research tasks use two targeted queries; simple factual
    # questions use one. No domain-specific genre branching.
    try:
        results = search_research_sources(
            query=query,
            detailed=research_task,
        )
    except Exception as exc:
        debug(
            "PIPELINE_ERROR",
            f"{type(exc).__name__}: {str(exc)[:180]}",
        )
        results = []

    source_urls = get_search_urls(results)
    search_evidence = format_results(results)

    if search_evidence:
        evidence = f"WEB SEARCH RESULTS:\n{search_evidence}"
    else:
        evidence = (
            "WEB SEARCH RESULTS:\n"
            "No usable search results were retrieved. "
            "Do not invent facts or citations."
        )

    # Extract at most one webpage for research tasks or explicit URLs.
    url_match = re.search(r"https?://[^\s<>\])]+", user_text)
    user_url = clean_url(url_match.group(0)) if url_match else ""

    wants_extract = research_task or bool(user_url)
    selected_url = ""

    if user_url and wants_extract:
        selected_url = user_url
    elif wants_extract:
        selected_url = choose_source_for_extraction(results)

    if wants_extract and selected_url:
        extracted = extract_webpage(selected_url, focus=query)
        evidence += f"\n\nEXTRACTED WEBPAGE:\n{extracted}"

        extracted_match = re.search(
            r"^Source URL:\s*(https?://\S+)",
            extracted,
            flags=re.MULTILINE,
        )

        if extracted_match:
            extracted_url = clean_url(extracted_match.group(1))

            if extracted_url not in source_urls:
                source_urls.append(extracted_url)

    source_urls = list(dict.fromkeys(source_urls))

    callback_context.state["temp:source_urls"] = source_urls

    debug("SOURCES", f"Captured {len(source_urls)} URLs")

    # Replace prior conversation history to keep the LLM input small.
    latest_message = llm_request.contents[user_index]

    llm_request.contents = [
        types.Content(
            role="user",
            parts=list(latest_message.parts or []) + [
                types.Part(text=f"\n\n{evidence}")
            ],
        )
    ]

    debug("PIPELINE", "History trimmed; compact evidence attached")

    return None


def after_model_callback(callback_context, llm_response):
    """Add citations and save completed research automatically."""

    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) or []

    answer = "\n".join(
        part.text
        for part in parts
        if getattr(part, "text", None)
    ).strip()

    if not answer:
        debug("POST", "No text response")
        return None

    source_urls = callback_context.state.get(
        "temp:source_urls",
        [],
    )

    answer = add_missing_sources(answer, source_urls)

    save_requested = callback_context.state.get(
        "temp:save_report",
        False,
    )

    # Keep the returned answer and saved report consistent.
    if content is not None:
        content.parts = [types.Part(text=answer)]

    if not save_requested:
        return llm_response

    if is_incomplete_answer(answer):
        debug("SAVE", "SKIPPED: response indicates incomplete research")
        return llm_response

    if not source_urls:
        debug("SAVE", "SKIPPED: no source URLs available")
        return llm_response

    title = callback_context.state.get(
        "temp:report_title",
        "Research Report",
    )

    saved_path = save_research_report(title, answer)

    if not saved_path.startswith("reports/"):
        debug("SAVE", "FAILED: report was not saved")
        return llm_response

    debug("SAVE", f"REPORT_CONFIRMED path={saved_path}")

    answer += f"\n\nSaved report: {saved_path}"

    if content is not None:
        content.parts = [types.Part(text=answer)]

    return llm_response


# ==========================================================
# 7. ROOT AGENT
# ==========================================================

root_agent = Agent(
    name="research_agent",

    model=LiteLlm(
        model="groq/qwen/qwen3.8-27b",
        reasoning_effort="none",
        reasoning_format="hidden",
        max_tokens=1200,
    ),

    instruction="""
You are a topic-independent, evidence-based research assistant.

The latest user message contains WEB SEARCH RESULTS and,
when extraction was requested, EXTRACTED WEBPAGE content.

You can research any subject. Do not use a fixed list of topics.

EVIDENCE RULES:
- Use the supplied evidence as your source of truth.
- Do not invent facts, dates, statistics, quotations, or URLs.
- Cite only URLs present in the evidence.
- Distinguish verified facts, reported claims, plans, and uncertainty.
- Prefer primary or official sources when the retrieved results support it.
- A URL being included does not itself prove a claim.

ANSWER RULES:
- For ordinary questions, answer concisely.
- For research, reports, comparisons, and analysis, use clear headings.
- Include key findings, limitations, and a Sources section.
- When requested, produce detailed reports using available evidence.
- State when the evidence is insufficient.
- Do not call additional tools; retrieval already occurred in Python.
- Do not repeat searches.

REPORT RULES:
- Research/report tasks are saved automatically by the Python pipeline.
- Include actual source URLs in the report.
- Never present an unsupported claim as verified.
""",

    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
)


debug("STARTUP", "Generic research pipeline configured")