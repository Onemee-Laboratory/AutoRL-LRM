#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Watchlist Manager for paper update

This script orchestrates loading the DB, adding new entries, 
and updating the Markdown watchlists for Emacs.
"""

import os
import sys
import json
import re
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable


# ---------------------------------------------------------------------------
# 0. Configura logging prō debugging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Ambiens Integratio (Stricta)
# ---------------------------------------------------------------------------
try:
    # Vallum Primum: Sine hoc dīrectōriō rādīcis, nihil agimus.
    AUTORL_HOME = Path(os.environ["AUTORL_HOME"])
    
    # Ceterae sēmitae ex env.sh petuntur
    AUTORL_DATA      = Path(os.environ["AUTORL_DATA"])
    WATCHLIST_DIR    = Path(os.environ["AUTORL_WATCHLIST"])
    PAPER_DB_JSON    = Path(os.environ["AUTORL_PAPERS_DB"])

except KeyError as e:
    print(f"ERROR: Variable {e} is not set.")
    print("Please run 'source env.sh' before executing this script.")
    sys.exit(1)

# Assevērāmus dīrectōria exstāre (sī env.sh ea nōn creāvit)
for d in [AUTORL_DATA, WATCHLIST_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class TierRule:
  """Defines a watchlist category without hard-coded strings."""
  id: str
  filename: str
  priority: int
  condition: Callable[[Dict[str, Any]], bool]

class WatchlistEngine:
  """Orchestrates paper distribution using a dynamic rule registry."""

  def __init__(self, config_path: Optional[Path] = None):
    self.rules: List[TierRule] = []
    self._load_rules_from_env()
    if config_path and config_path.exists():
      self._load_rules_from_file(config_path)

  def _load_rules_from_env(self):
    """Hydrates rules from environment variables (The Source of Truth)."""
    current_year = datetime.now().year
    
    # 1. NEW RESEARCH TIER
    self.add_rule(TierRule(
        id="news",
        filename=os.getenv("AUTORL_WATCHLIST_NEW_PAPERS", "new_papers.md"),
        priority=10,
        condition=lambda p: p.get("year", 0) >= int(os.getenv("AUTORL_WATCHLIST_YEAR_LIMIT", 2025))
    ))

    # 2. CLASSIC TIER
    self.add_rule(TierRule(
        id="classic",
        filename=os.getenv("AUTORL_WATCHLIST_CLASSIC_PAPERS", "classic_papers.md"),
        priority=5,  # Higher priority (lower number) for classics
        condition=lambda p: p.get("classic", False) is True
    ))

    # 3. HIGH INDEX TIER
    self.add_rule(TierRule(
        id="high_index",
        filename=os.getenv("AUTORL_WATCHLIST_HIGH_INDEX_PAPERS", "high_index_papers.md"),
        priority=20,
        condition=lambda p: p.get("citations", 0) >= int(os.getenv("AUTORL_WATCHLIST_CITE_LIMIT", 100))
    ))

  def add_rule(self, rule: TierRule):
    """Adds a rule and maintains priority sorting."""
    self.rules.append(rule)
    self.rules.sort(key=lambda r: r.priority)

  def route_paper(self, paper: Dict[str, Any]) -> str:
    """Finds the first matching rule for a paper."""
    for rule in self.rules:
      if rule.condition(paper):
        return rule.filename
    return os.getenv("AUTORL_WATCHLIST_FILE_DEFAULT", "misc_papers.md")

# ---------------------------------------------------------------------------
# 2. Metadata Storage
# ---------------------------------------------------------------------------
def load_db(path: Path) -> Dict[str, Any]:
    if not path.exists():
        logger.info("Database non inventa; nova creātur: %s", path)
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_db(path: Path, db: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(db, f, indent=2)

# ---------------------------------------------------------------------------
# 3. Ratio Scōpi (Reasoning Engine)
# ---------------------------------------------------------------------------
def score_paper(paper):
    """Calculat pondus lōgicum prō RL/LRM research."""
    score = 0.0
    
    # A. Status Primarius
    if paper.get("classic"): score += 10.0  # Gravitās maxima
    if paper.get("news"):    score += 2.0   # Incrementum novitātis
    
    # B. Citātiōnēs (Logarithmica fōrma melior est prō outliers)
    score += paper.get("citations", 0) * 0.05
    
    # C. Decay Temporālis (In LLM/RL, 1 annus est aevum)
    year   = paper.get("year", datetime.now().year)
    age    = datetime.now().year - year
    score += max(0, 5 - age * 1.0)  # Celerior dēminūtiō prō LRM
    
    # D. Keyword Boost (RLVR, GRPO, Reasoning)
    title    = paper.get("title", "").lower()
    keywords = ["reasoning", "rlvr", "grpo", "verifiable", "alignment", "scaling"]
    if any(k in title for k in keywords):
        score += 4.0
        
    return round(score, 2)

# ---------------------------------------------------------------------------
# 4. Watchlist Update (Markdown Tables for Emacs)
# ---------------------------------------------------------------------------
def update_watchlists():
    # Recalculāre omnēs scōpōs et pūrgāre nōmina
    for pid in PAPER_DB:
        PAPER_DB[pid]["score"] = score_paper(PAPER_DB[pid])
    
    ranked_papers = sorted(PAPER_DB.values(), key=lambda x: x["score"], reverse=True)

    watchlist_map = {
        "classic_papers.md": [],
        "high_index_papers.md": [],
        "old_papers.md": [],
        "news_papers.md": []
    }

    # Distribūtiō chartārum secundum condiciōnēs prīstinas
    current_year = datetime.now().year
    for paper in ranked_papers:
        if paper.get("classic", False):
            watchlist_map["classic_papers.md"].append(paper)
        elif paper.get("citations", 0) > 100:
            watchlist_map["high_index_papers.md"].append(paper)
        elif paper.get("year", 0) < current_year - 5:
            watchlist_map["old_papers.md"].append(paper)
        elif paper.get("news", False) or paper.get("year", 0) >= 2025:
            watchlist_map["news_papers.md"].append(paper)

    # Scrīptūra in fōrmā "bullet blocks" (nōn tabulārum)
    for fname, papers in watchlist_map.items():
        path      = WATCHLIST_DIR / fname
        tz        = timezone(timedelta(hours=-7))
        timestamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M %Z")
        
        with open(path, "w") as f:
            f.write(f"\n\n---\n## Watchlist: {fname} | {timestamp}\n\n")
            f.write(f"Found {len(papers)} papers.\n\n")
            
            for p in papers:
                tags  = p.get("tags", ["general"])
                score = p.get("score", 0)
                f.write(f"### {p.get('title', 'Unknown')}\n")
                f.write(f"- **Source:** watchlist  |  **ID:** {p.get('id', '')}\n")
                f.write(f"- **Authors:** {p.get('authors', '')}\n")
                f.write(f"- **Published:** {p.get('year', '')}\n")
                f.write(f"- **URL:** {p.get('url', '')}\n")
                f.write(f"- **Relevance score:** {score}  |  **Tags:** {', '.join(tags)}\n")
                ab = p.get("abstract", "")[:400]
                f.write(f"- **Abstract:** {ab}\n\n")
                
        print(f"✅ Updated {fname} with {len(papers)} papers")

    print(f"✅ Watchlists rēctē mūtātae in {WATCHLIST_DIR}")


    
    
# ---------------------------------------------------------------------------
# 4. Watchlist Update (Markdown Tables for Emacs)
# ---------------------------------------------------------------------------    
def update_watchlists(paper_db: Dict[str, Any], output_dir: Path):
  """Updates all watchlists using the dynamic engine.
  
  Args:
    paper_db: Full database of papers.
    output_dir: Path where markdown files are saved.
  """
  engine = WatchlistEngine()
  buckets: Dict[str, List[Dict]] = {}

  # 1. Routing
  for paper in paper_db.values():
    target_file = engine.route_paper(paper)
    if target_file not in buckets:
      buckets[target_file] = []
    buckets[target_file].append(paper)

  # 2. Writing (Scalable Loop)
  for filename, papers in buckets.items():
    full_path = output_dir / filename
    write_watchlist_file(full_path, papers)
    logger.info("Watchlist %s renovāta: %d chartae.", filename, len(papers))

def write_watchlist_file(path: Path, papers: List[Dict]):
  """Low-level file writer with standardized format."""
  tz_offset = int(os.getenv("AUTORL_TZ_HOUR", -7))
  # Logic for writing bullet blocks...
  pass
    
# ---------------------------------------------------------------------------
# 5. Add Paper Function (Cum plūribus fieldīs)
# ---------------------------------------------------------------------------
def add_paper(title, year=None, citations=0, classic=False, news=False, authors="Unknown", url="", abstract="", tags=None):
    pid = re.sub(r'\W+', '_', title.lower()).strip('_')
    PAPER_DB[pid] = {
        "id": pid,
        "title": title,
        "year": year or datetime.now().year,
        "citations": citations,
        "classic": classic,
        "news": news,
        "authors": authors,
        "url": url,
        "abstract": abstract,
        "tags": tags or ["general"],
        "score": 0
    }

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    """Orchestrates the data flow from JSON to Markdown."""
    
    # 1. Trahe sēmitās (Paths) ex env.sh (Zero Hard-Code)
    db_env = os.getenv("AUTORL_PAPERS_DB")
    wl_env = os.getenv("AUTORL_WATCHLIST")
    
    if not db_env or not wl_env:
        logger.error("Ambūlācrum non dēfīnītum. Fac 'source env.sh'.")
        sys.exit(1)

    db_path = Path(db_env)
    output_dir = Path(wl_env)

    # 2. Lēctiō (Load State)
    # Nota: load_db() dēbet PAPER_DB dēfīnīre
    global PAPER_DB
    PAPER_DB = load_db(db_path)
    initial_count = len(PAPER_DB)

    # 3. Additiō (Exempla vel Logic Scannandi)
    # In 'production', hīc legimus nova dāta ex scan_papers.py
    add_paper("DeepMind AlphaGo", 2016, 5000, classic=True, 
              authors="Silver et al.", url="https://nature.com/articles/nature16961")
    add_paper("DeepSeek-R1", 2025, 150, news=True, 
              authors="DeepSeek-AI", abstract="Reinforcement Learning for reasoning...")

    # 4. Appellātiō Rēctificāta
    # Trānsferimus PAPER_DB et Path dīrectē
    update_watchlists(paper_db=PAPER_DB, output_dir=output_dir)

    # 5. Sērvātiō (Save State)
    if len(PAPER_DB) > initial_count:
        save_db(db_path, PAPER_DB)
        logger.info("✅ Database renovāta: %d entries.", len(PAPER_DB))
        # Exit 0 significat 'Nova dāta scripta sunt'
        sys.exit(0)
    else:
        logger.info("⚠️ Nihil novī ad addendum.")
        # Exit 1 (vel code specialis) significat 'Nihil mūtātum'
        sys.exit(1)

if __name__ == "__main__":
    main()    
