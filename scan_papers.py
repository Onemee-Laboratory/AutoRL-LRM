# -*- coding: utf-8 -*-
"""
scan_papers.py — Literature scanner for AutoRL-LRM self-improvement loop
=========================================================================
Part of the Meta Loop (autoloop_meta.sh). 
Dual-Engine: ArXiv (Pre-prints) + Semantic Scholar (Conferences/Journals).

Usage Examples:
    python scan_papers.py --since 7 --queries "RLVR"
    python scan_papers.py --authors "Yoshua Bengio" --keywords "reasoning"

Output: $AUTORL_WATCHLIST/new_papers.md (Bullet Block Format)
"""


import argparse
import datetime
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any

def get_env_config(name: str) -> str:
  """Retrieves a mandatory environment variable or exits."""
  val = os.environ.get(name)
  if not val:
    print(f"❌ ERROR: Variable '{name}' is NOT SET in env.sh.")
    sys.exit(1)
  return val


def robust_request(url: str, max_retries: int = 3) -> bytes:
  """Performs an HTTP request with exponential backoff for 429 errors."""
  headers = {"User-Agent": "AutoRL-Scanner/2.0"}
  for attempt in range(max_retries):
    try:
      req = urllib.request.Request(url, headers=headers)
      with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()
    except urllib.error.HTTPError as e:
      if e.code == 429:
        wait = int(e.headers.get("Retry-After", 60))
        print(f"  ⚠️ [429] Rate limit hit. Waiting {wait}s...")
        time.sleep(wait)
      else:
        break
    except Exception:
      time.sleep(5)
  return None


def fetch_arxiv(query: str, prefix: str = "all:", limit: int = 10) -> List[Dict]:
  """Fetches papers from the ArXiv API."""
  if not query:
    return []
  encoded_q = urllib.parse.quote(f"{prefix}\"{query.strip()}\"")
  url = (f"https://export.arxiv.org/api/query?search_query={encoded_q}"
         f"&max_results={limit}&sortBy=submittedDate&sortOrder=descending")
  data = robust_request(url)
  if not data:
    return []

  try:
    root = ET.fromstring(data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("atom:entry", ns):
      aid = entry.findtext("atom:id", "", ns).split("/abs/")[-1].strip()
      results.append({
          "id": aid,
          "source": "arxiv",
          "title": entry.findtext("atom:title", "", ns).strip().replace("\n", " "),
          "abstract": entry.findtext("atom:summary", "", ns).strip().replace("\n", " "),
          "authors": ", ".join([a.findtext("atom:name", "", ns) 
                               for a in entry.findall("atom:author", ns)]),
          "year": entry.findtext("atom:published", "", ns)[:4],
          "date_sort": entry.findtext("atom:published", "", ns)[:10].replace("-", ""),
          "url": f"https://arxiv.org/abs/{aid}",
          "tags": ["arxiv"]
      })
    return results
  except ET.ParseError:
    return []


def fetch_s2(query: str, limit: int = 10) -> List[Dict]:
  """Fetches papers from the Semantic Scholar API."""
  if not query:
    return []
  params = urllib.parse.urlencode({
      "query": query,
      "limit": limit,
      "fields": "paperId,title,abstract,authors,year,publicationDate,url"
  })
  url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
  data = robust_request(url)
  if not data:
    return []

  try:
    items = json.loads(data).get("data", [])
    return [{
        "id": p.get("paperId"),
        "source": "s2",
        "title": p.get("title", ""),
        "abstract": p.get("abstract") or "No abstract available.",
        "authors": ", ".join([a.get("name") for a in p.get("authors", [])]),
        "year": str(p.get("year", "")),
        "date_sort": (p.get("publicationDate") or 
                     f"{p.get('year')}-01-01").replace("-", ""),
        "url": p.get("url"),
        "tags": ["semantic_scholar"]
    } for p in items]
  except json.JSONDecodeError:
    return []


def perform_scan(queries: List[str], authors: List[str], 
                 since_days: int, max_per_q: int) -> Dict[str, Dict]:
  """Orchestrates data collection with CLI status updates."""
  found_papers = {}
  since_dt = (datetime.datetime.now() - 
              datetime.timedelta(days=since_days)).strftime("%Y%m%d")

  print(f"📡 Starting scan for: {queries} {authors}")

  for author in authors:
    for p in fetch_arxiv(author, prefix="au:", limit=max_per_q):
      if p["date_sort"] >= since_dt:
        found_papers[p["id"]] = p
    time.sleep(1)

  for q in queries:
    combined = fetch_arxiv(q, limit=max_per_q) + fetch_s2(q, limit=max_per_q)
    for p in combined:
      if p["date_sort"] >= since_dt:
        found_papers[p["id"]] = p
    time.sleep(1)

  return found_papers


def write_output(papers: List[Dict], meta: Dict[str, Any]):
  """Writes final results and prints success message."""
  watchlist_dir = Path(get_env_config("AUTORL_WATCHLIST"))
  filename = get_env_config("AUTORL_WATCHLIST_NEW_PAPERS")
  output_path = watchlist_dir / filename
  
  tz = datetime.timezone(datetime.timedelta(hours=-7))
  timestamp = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M %Z")

  watchlist_dir.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"\n\n---\n## Watchlist: {filename} | {timestamp}\n\n")
    f.write("### # A. Scrutāmur (Scanning Context)\n")
    f.write(f"- **Lookback:** {meta['since']} days\n")
    f.write(f"- **Entities:** {', '.join(meta['entities'])}\n\n---\n\n")

    for p in papers:
      f.write(f"### {p.get('title')}\n")
      f.write(f"- **Source:** {p.get('source')} | **ID:** {p.get('id')}\n")
      f.write(f"- **Authors:** {p.get('authors')}\n")
      f.write(f"- **URL:** {p.get('url')}\n")
      f.write(f"- **Relevance score:** {p.get('score')}  |  **Tags:** arxiv\n")
      f.write(f"- **Abstract:** {p.get('abstract')[:400].replace(os.linesep, ' ')}...\n\n")
  
  print(f"✅ Success: Results saved to {output_path}")


def main():
  """Parses CLI arguments and executes the research pipeline."""
  parser = argparse.ArgumentParser(description="AutoRL-LRM Literature Scanner")
  parser.add_argument("--queries", nargs="+", default=[])
  parser.add_argument("--authors", nargs="+", default=[])
  parser.add_argument("--institutes", nargs="+", default=["Allen Institute"])
  parser.add_argument("--universities", nargs="+", default=["MIT"])
  parser.add_argument("--companies", nargs="+", default=["DeepMind"])
  parser.add_argument("--keywords", nargs="+", default=["Reasoning", "RL"])
  parser.add_argument("--since", type=int, default=7)
  parser.add_argument("--max", type=int, default=10)
  args = parser.parse_args()

  # Warning and Default Scope
  if not args.queries and not args.authors:
    print("⚠️ WARNING: No --queries or --authors provided.")
    print("💡 Using default AutoRL-LRM research scope...")
    args.queries = [
    "RLVR mathematical reasoning",
    "GRPO reward shaping math LLM",
    "REINFORCE RLOO DAPO math reasoning",
    "KL divergence policy collapse math",
    "reward hacking verifier math language model",
]

  entities = args.institutes + args.universities + args.companies
  search_queries = list(args.queries)
  for entity in entities:
    search_queries.append(f"\"{entity}\" reasoning")

  raw_data = perform_scan(search_queries, args.authors, args.since, args.max)
  
  # Filter and Sort
  kws = [k.lower() for k in args.keywords]
  valid_papers = []
  for p in raw_data.values():
    score = sum(1 for k in kws if k in (p["title"] + p["abstract"]).lower())
    if score > 0 or not kws:
      p["score"] = score
      valid_papers.append(p)
  
  if valid_papers:
    sorted_papers = sorted(valid_papers, key=lambda x: x.get("score", 0), reverse=True)
    write_output(sorted_papers, {"since": args.since, "entities": entities})
  else:
    print("ℹ️ No relevant papers found.")


if __name__ == "__main__":
  main()
