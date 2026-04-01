"""
scan_papers.py — Literature scanner for AutoRL-LRM self-improvement loop
=========================================================================
Queries arxiv and Semantic Scholar for new papers relevant to RLVR,
reward shaping, and large reasoning models. Extracts actionable
hyperparameters and reward functions. Writes new_papers.md.

Usage:
    python scan_papers.py                        # scan last 7 days
    python scan_papers.py --since 14             # scan last 14 days
    python scan_papers.py --since 30 --max 50    # broader sweep

Output:
    $AUTORL_HOME/new_papers.md   (appended, timestamped)

Dependencies: requests (stdlib-adjacent, always available)
"""

import os
import re
import json
import time
import argparse
import datetime
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AUTORL_HOME = os.environ.get("AUTORL_HOME", str(Path(__file__).resolve().parent))
OUTPUT_PATH = os.path.join(AUTORL_HOME, "new_papers.md")

# Search queries — tuned for RLVR + LRM + reward shaping
ARXIV_QUERIES = [
    "reinforcement learning verifiable rewards reasoning",
    "GRPO reward shaping language model math",
    "RLVR large reasoning model reward hacking",
    "KL divergence policy collapse language model training",
    "reward shaping chain of thought verifier",
    "test time compute scaling reasoning model",
    "MCMC sampling language model inference",
]

S2_QUERIES = [
    "RLVR reward shaping reasoning model 2025 2026",
    "GRPO KL penalty reward hacking LLM",
    "chain of thought verifier learnability",
    "escape radius policy divergence reinforcement learning",
]

# Keywords that flag a paper as highly relevant
HIGH_RELEVANCE_KEYWORDS = [
    "RLVR", "GRPO", "PPO", "reward shaping", "verifiable reward",
    "chain of thought", "CoT verifier", "reasoning model", "escape radius",
    "KL divergence", "policy collapse", "diversity collapse", "pass@k",
    "CGDB", "deviation bonus", "power sampling", "Metropolis",
    "Hessian", "loss landscape", "leash", "math reasoning",
]

# Keywords for extracting actionable parameters from abstracts
PARAM_PATTERNS = [
    (r"learning rate[s]?\s+(?:of\s+)?(\d[\d\.e\-]+)", "LR"),
    (r"temperature\s+(?:of\s+)?(\d[\d\.]+)",           "TEMPERATURE"),
    (r"KL\s+(?:coefficient|coeff|weight)\s+(?:of\s+)?(\d[\d\.e\-]+)", "KL_COEFF"),
    (r"(\d+)\s+samples?\s+per\s+prompt",               "N_SAMPLES"),
    (r"group\s+size\s+(?:of\s+)?(\d+)",                "N_SAMPLES"),
    (r"(\d+)\s+rollouts?",                             "N_SAMPLES"),
    (r"alpha\s*[=:]\s*(\d[\d\.]+)",                    "POWER_ALPHA"),
]


# ---------------------------------------------------------------------------
# arxiv API
# ---------------------------------------------------------------------------

ARXIV_NS = "http://www.w3.org/2005/Atom"

def search_arxiv(query: str, since_days: int, max_results: int = 20) -> list[dict]:
    """Query arxiv search API. Returns list of paper dicts."""
    since_dt = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - datetime.timedelta(days=since_days)
    since_str = since_dt.strftime("%Y%m%d")

    params = urllib.parse.urlencode({
        "search_query": f"all:{urllib.parse.quote(query)}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"https://export.arxiv.org/api/query?{params}"

    max_attempts = 3
    xml_data = None
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                xml_data = resp.read().decode("utf-8")
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (attempt + 1)
                print(f"  [arxiv] rate limited (429) for '{query}' — waiting {wait}s (attempt {attempt+1}/{max_attempts})")
                time.sleep(wait)
                if attempt == max_attempts - 1:
                    print(f"  [arxiv] giving up after {max_attempts} attempts for '{query}'")
                    return []
            else:
                print(f"  [arxiv] HTTP error {e.code} for '{query}': {e}")
                return []
        except Exception as e:
            print(f"  [arxiv] request failed for '{query}': {e}")
            return []
 
    if xml_data is None:
        return []

    papers = []
    try:
        root = ET.fromstring(xml_data)
        for entry in root.findall(f"{{{ARXIV_NS}}}entry"):
            published = entry.findtext(f"{{{ARXIV_NS}}}published", "")
            # filter by date
            pub_date = published[:10].replace("-", "")
            if pub_date < since_str:
                continue

            arxiv_id = entry.findtext(f"{{{ARXIV_NS}}}id", "").split("/abs/")[-1].strip()
            title   = entry.findtext(f"{{{ARXIV_NS}}}title", "").replace("\n", " ").strip()
            summary = entry.findtext(f"{{{ARXIV_NS}}}summary", "").replace("\n", " ").strip()
            authors = [
                a.findtext(f"{{{ARXIV_NS}}}name", "")
                for a in entry.findall(f"{{{ARXIV_NS}}}author")
            ]
            papers.append({
                "source":    "arxiv",
                "id":        arxiv_id,
                "title":     title,
                "abstract":  summary,
                "authors":   authors[:4],
                "published": published[:10],
                "url":       f"https://arxiv.org/abs/{arxiv_id}",
            })
    except ET.ParseError as e:
        print(f"  [arxiv] parse error: {e}")

    return papers


# ---------------------------------------------------------------------------
# Semantic Scholar API
# ---------------------------------------------------------------------------

def search_s2(query: str, since_days: int, max_results: int = 10) -> list[dict]:
    """Query Semantic Scholar public API."""
    since_year = (datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - datetime.timedelta(days=since_days)).year

    params = urllib.parse.urlencode({
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,authors,year,externalIds,publicationDate",
        "year": f"{since_year}-",
    })
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"

    
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AutoRL-LRM-Scanner/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break   # success — exit retry loop
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (attempt + 1)   # 5s, 10s, 15s
                print(f"  [S2] rate limited (429) for '{query}' — waiting {wait}s (attempt {attempt+1}/{max_attempts})")
                time.sleep(wait)
                if attempt == max_attempts - 1:
                    print(f"  [S2] giving up after {max_attempts} attempts for '{query}'")
                    return []
            else:
                print(f"  [S2] HTTP error {e.code} for '{query}': {e}")
                return []
        except Exception as e:
            print(f"  [S2] request failed for '{query}': {e}")
            return []

    papers = []
    for p in data.get("data", []):
        ext_ids = p.get("externalIds", {})
        arxiv_id = ext_ids.get("ArXiv", "")
        papers.append({
            "source":    "semanticscholar",
            "id":        arxiv_id or p.get("paperId", ""),
            "title":     p.get("title", ""),
            "abstract":  (p.get("abstract") or "")[:800],
            "authors":   [a["name"] for a in p.get("authors", [])[:4]],
            "published": p.get("publicationDate", str(p.get("year", ""))),
            "url":       (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id
                         else f"https://www.semanticscholar.org/paper/{p.get('paperId','')}"),
        })
    return papers


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

def relevance_score(paper: dict) -> int:
    text = (paper["title"] + " " + paper["abstract"]).lower()
    score = sum(1 for kw in HIGH_RELEVANCE_KEYWORDS if kw.lower() in text)
    return score


def extract_params(abstract: str) -> dict:
    """Try to extract concrete hyperparameter values mentioned in the abstract."""
    found = {}
    for pattern, name in PARAM_PATTERNS:
        m = re.search(pattern, abstract, re.IGNORECASE)
        if m:
            found[name] = m.group(1)
    return found


def classify_contribution(abstract: str) -> list[str]:
    """Tag what kind of contribution this paper makes."""
    tags = []
    ab = abstract.lower()
    if any(w in ab for w in ["reward shaping", "reward function", "reward signal"]):
        tags.append("reward_shaping")
    if any(w in ab for w in ["sampling", "inference", "test-time", "test time"]):
        tags.append("inference_method")
    if any(w in ab for w in ["kl", "divergence", "constraint", "penalty"]):
        tags.append("kl_regularisation")
    if any(w in ab for w in ["diversity", "collapse", "pass@k", "pass@"]):
        tags.append("diversity")
    if any(w in ab for w in ["hessian", "curvature", "sharpness", "landscape"]):
        tags.append("loss_landscape")
    if any(w in ab for w in ["verifier", "critic", "discriminator"]):
        tags.append("verifier")
    if any(w in ab for w in ["curriculum", "difficulty", "progressive"]):
        tags.append("curriculum")
    if any(w in ab for w in ["policy gradient", "advantage estimator", "reinforce",
                              "rloo", "dapo", "grpo", "ppo", "actor-critic",
                              "update rule", "policy optimization", "policy optimisation"]):
        tags.append("algorithm")
    return tags or ["general"]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def load_seen_ids(output_path: str) -> set:
    seen = set()
    if not os.path.exists(output_path):
        return seen
    with open(output_path) as f:
        for line in f:
            m = re.search(r"arxiv\.org/abs/([\d\.v]+)", line)
            if m:
                seen.add(m.group(1))
            m2 = re.search(r"\*\*ID:\*\*\s*([\w\.]+)", line)
            if m2:
                seen.add(m2.group(1))
    return seen


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def format_paper(paper: dict, score: int, params: dict, tags: list) -> str:
    lines = [
        f"### {paper['title']}",
        f"- **Source:** {paper['source']}  |  **ID:** {paper['id']}",
        f"- **Authors:** {', '.join(paper['authors'])}",
        f"- **Published:** {paper['published']}",
        f"- **URL:** {paper['url']}",
        f"- **Relevance score:** {score}  |  **Tags:** {', '.join(tags)}",
    ]
    if params:
        param_str = ", ".join(f"`{k}={v}`" for k, v in params.items())
        lines.append(f"- **Extracted params:** {param_str}")

    # truncate abstract
    ab = paper["abstract"]
    if len(ab) > 400:
        ab = ab[:397] + "..."
    lines.append(f"- **Abstract:** {ab}")
    lines.append("")
    return "\n".join(lines)


def write_output(papers: list[dict], output_path: str, since_days: int):
    timestamp = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M UTC")
    header = (
        f"\n\n---\n"
        f"## Scan: {timestamp}  (last {since_days} days)\n\n"
        f"Found {len(papers)} relevant papers.\n\n"
    )

    sections = {"algorithm": [], "reward_shaping": [], "inference_method": [],
                "kl_regularisation": [], "diversity": [], "loss_landscape": [],
                "verifier": [], "curriculum": [], "general": []}

    for p in papers:
        tags  = classify_contribution(p["abstract"])
        score = relevance_score(p)
        prms  = extract_params(p["abstract"])
        block = format_paper(p, score, prms, tags)
        for tag in tags:
            sections.get(tag, sections["general"]).append(block)

    content = header
    for section, blocks in sections.items():
        if blocks:
            content += f"### Category: {section}\n\n"
            content += "\n".join(blocks)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        f.write(content)

    print(f"\n[scan_papers] Written {len(papers)} papers → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(since_days: int = 7, max_per_query: int = 15, min_score: int = 2):
    print(f"[scan_papers] Scanning last {since_days} days ...")
    seen = load_seen_ids(OUTPUT_PATH)
    all_papers: dict[str, dict] = {}   # id → paper (dedup)

    # arxiv
    for i, query in enumerate(ARXIV_QUERIES):
        print(f"  [arxiv {i+1}/{len(ARXIV_QUERIES)}] {query[:60]}")
        results = search_arxiv(query, since_days, max_per_query)
        for p in results:
            if p["id"] and p["id"] not in seen and p["id"] not in all_papers:
                all_papers[p["id"]] = p
        time.sleep(1.5)   # arxiv rate limit: 3 req/s recommended

    # Semantic Scholar
    for i, query in enumerate(S2_QUERIES):
        print(f"  [S2 {i+1}/{len(S2_QUERIES)}] {query[:60]}")
        results = search_s2(query, since_days, max_per_query)
        for p in results:
            if p["id"] and p["id"] not in seen and p["id"] not in all_papers:
                all_papers[p["id"]] = p
        time.sleep(1.0)

    # Score and filter
    scored = []
    for paper in all_papers.values():
        score = relevance_score(paper)
        if score >= min_score:
            scored.append((score, paper))

    scored.sort(key=lambda x: -x[0])
    filtered = [p for _, p in scored]

    print(f"\n[scan_papers] {len(all_papers)} total → {len(filtered)} above threshold (score≥{min_score})")

    if filtered:
        write_output(filtered, OUTPUT_PATH, since_days)
    else:
        print("[scan_papers] No new relevant papers found.")

    return filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--since",     type=int, default=7,  help="Days to look back")
    parser.add_argument("--max",       type=int, default=15, help="Max results per query")
    parser.add_argument("--min_score", type=int, default=2,  help="Min relevance score to include")
    args = parser.parse_args()
    run(since_days=args.since, max_per_query=args.max, min_score=args.min_score)
