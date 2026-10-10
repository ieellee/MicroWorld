import os
import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from openai import OpenAI

URL   = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
KEY   = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4")

client = OpenAI(base_url=URL, api_key=KEY)

SUPPORT_DIR   = Path(__file__).parent / "support"
KG_DIR        = SUPPORT_DIR / "KG"
NODES_PATH    = KG_DIR / "nodes.json"
DESC_OUT_PATH = SUPPORT_DIR / "entity_descriptions.json"

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MIN_DESC_LEN = 40          # LLM fallback


def _ncbi_get(url: str, timeout: int = 12) -> dict:
    """HTTP GET →  JSON， dict。"""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"    [NCBI ] {e}")
        return {}


def ncbi_esearch(db: str, term: str, api_key: str = "", retmax: int = 3) -> list[str]:
    """ NCBI  term， UID 。"""
    params = {"db": db, "term": term, "retmax": str(retmax), "retmode": "json"}
    if api_key:
        params["api_key"] = api_key
    url = f"{NCBI_BASE}/esearch.fcgi?" + urllib.parse.urlencode(params)
    data = _ncbi_get(url)
    return data.get("esearchresult", {}).get("idlist", [])


def ncbi_esummary(db: str, uid: str, api_key: str = "") -> dict:
    """ NCBI （JSON ）， dict。"""
    params = {"db": db, "id": uid, "retmode": "json"}
    if api_key:
        params["api_key"] = api_key
    url = f"{NCBI_BASE}/esummary.fcgi?" + urllib.parse.urlencode(params)
    data = _ncbi_get(url)
    result = data.get("result", {})
    return result.get(uid, {})


# ─── MeSH ───────────────────────────────────────────────────────────────────

def _verify_mesh_match(term: str, scope_note: str) -> bool:
    """
    ：MeSH scope note 。
    ：（≥5 ） scope note 。
    """
    #  ≥4 
    keywords = [w.lower() for w in re.split(r"\W+", term) if len(w) >= 4]
    if not keywords:
        return True   # ， NCBI
    note_lower = scope_note.lower()
    return any(kw in note_lower for kw in keywords)


def lookup_mesh(term: str, api_key: str = "", sleep: float = 0.35) -> dict | None:
    """
     MeSH， {"summary", "ncbi_id", "ncbi_db"}  None。

    （→）：
      1. {term}[mh]  —  MeSH Heading（），
      2. {term}[nm]  —  Name / 
      3. {term}      — 

     _verify_mesh_match ，。
    """
    queries = [f"{term}[mh]", f"{term}[nm]", term]

    for query in queries:
        uids = ncbi_esearch("mesh", query, api_key, retmax=1)
        time.sleep(sleep)
        if not uids:
            continue

        rec = ncbi_esummary("mesh", uids[0], api_key)
        time.sleep(sleep)

        scope   = rec.get("ds_scopenote", "").strip()
        mesh_ui = rec.get("ds_meshui", uids[0])

        if not scope or len(scope) < MIN_DESC_LEN:
            continue
        if not _verify_mesh_match(term, scope):
            continue   # ，

        return {
            "summary":  scope,
            "ncbi_db":  "mesh",
            "ncbi_id":  mesh_ui,
        }

    return None


# ─── Taxonomy ───────────────────────────────────────────────────────────────

def lookup_taxonomy(term: str, api_key: str = "", sleep: float = 0.35) -> dict | None:
    RANK_PRIORITY = {"species": 0, "genus": 1, "subspecies": 2,
                     "family": 3, "order": 4, "class": 5}

    for query in [f"{term}[orgn]", term]:
        uids = ncbi_esearch("taxonomy", query, api_key, retmax=5)
        time.sleep(sleep)
        if uids:
            break
    if not uids:
        return None

    #  summary， rank 
    candidates: list[tuple[int, dict]] = []
    for uid in uids[:5]:
        rec = ncbi_esummary("taxonomy", uid, api_key)
        time.sleep(sleep)
        if not rec:
            continue
        rank     = rec.get("rank", "").lower()
        priority = RANK_PRIORITY.get(rank, 99)
        candidates.append((priority, rec))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    best_rec  = candidates[0][1]
    sci_name  = best_rec.get("scientificname", "")
    common    = best_rec.get("commonname", "")
    rank      = best_rec.get("rank", "unknown")
    lineage   = best_rec.get("lineage", "")
    uid_found = best_rec.get("uid", uids[0])

    parts = []
    if sci_name:
        parts.append(f"{sci_name} is a {rank}.")
    if common:
        parts.append(f"Common name: {common}.")
    if lineage:
        lineage_short = "; ".join(lineage.rstrip("; ").split("; ")[-6:])
        parts.append(f"Taxonomy: {lineage_short}.")
    summary = " ".join(parts)

    if len(summary) < MIN_DESC_LEN:
        return None
    return {
        "summary":  summary,
        "ncbi_db":  "taxonomy",
        "ncbi_id":  uid_found,
    }



def lookup_gene(term: str, api_key: str = "", sleep: float = 0.35) -> dict | None:

    query = f"{term}[gene] AND Homo sapiens[orgn]"
    uids = ncbi_esearch("gene", query, api_key, retmax=1)
    time.sleep(sleep)
    if not uids:
        # 
        uids = ncbi_esearch("gene", term, api_key, retmax=1)
        time.sleep(sleep)
    if not uids:
        return None

    rec = ncbi_esummary("gene", uids[0], api_key)
    time.sleep(sleep)
    gene_summary = rec.get("summary", "").strip()
    gene_desc    = rec.get("description", "").strip()

    if gene_summary and len(gene_summary) >= MIN_DESC_LEN:
        text = gene_summary
    elif gene_desc:
        text = gene_desc
    else:
        return None
    return {
        "summary":  text,
        "ncbi_db":  "gene",
        "ncbi_id":  uids[0],
    }


# ───  node_type  NCBI  ─────────────────────────────────────────────

def ncbi_lookup(node: dict, api_key: str = "", sleep: float = 0.35) -> dict | None:
    """
     NCBI ：
      organism  → Taxonomy → MeSH
      structure → MeSH → Gene
      method    → MeSH
      condition → MeSH
      property  → None（ LLM）
    """
    term  = node["canonical_name"]
    ntype = node["node_type"]

    if ntype == "property":
        return None    # NCBI ， LLM

    if ntype == "organism":
        result = lookup_taxonomy(term, api_key, sleep)
        if result:
            return result
        return lookup_mesh(term, api_key, sleep)

    if ntype == "structure":
        result = lookup_mesh(term, api_key, sleep)
        if result:
            return result
        return lookup_gene(term, api_key, sleep)

    if ntype in ("method", "condition"):
        return lookup_mesh(term, api_key, sleep)

    return None


# ═══════════════════════════════════════════════════════════════════════════
# LLM （ NCBI ）
# ═══════════════════════════════════════════════════════════════════════════

LLM_PROMPTS = {
    "structure": (
        "You are a biomedical expert in microscopy and cell biology. "
        "Provide a structured JSON description for the biological structure below.\n\n"
        "Entity: {name}\nAliases: {aliases}\n"
        "Example captions:\n{context}\n\n"
        'Return JSON with keys: "summary", "microscopy_appearance", "biological_role", "related_methods".\n'
        "Return ONLY the JSON."
    ),
    "organism": (
        "You are a microbiology/virology expert. "
        "Provide a structured JSON description for the organism below.\n\n"
        "Organism: {name}\nAliases: {aliases}\n"
        "Example captions:\n{context}\n\n"
        'Return JSON with keys: "summary", "microscopy_appearance", "biological_role", "related_methods".\n'
        "Return ONLY the JSON."
    ),
    "method": (
        "You are a bioimaging expert. "
        "Provide a structured JSON description for the imaging method below.\n\n"
        "Method: {name}\nAliases: {aliases}\n"
        "Example captions:\n{context}\n\n"
        'Return JSON with keys: "summary", "principle", "resolution", "applications".\n'
        "Return ONLY the JSON."
    ),
    "condition": (
        "You are a cell biology expert. "
        "Provide a structured JSON description for the experimental condition below.\n\n"
        "Condition: {name}\nAliases: {aliases}\n"
        "Example captions:\n{context}\n\n"
        'Return JSON with keys: "summary", "biological_significance".\n'
        "Return ONLY the JSON."
    ),
    "property": (
        "You are a microscopy image analysis expert. "
        "Provide a structured JSON description for the morphological property below.\n\n"
        "Property: {name}\nAliases: {aliases}\n"
        "Example captions:\n{context}\n\n"
        'Return JSON with keys: "summary", "biological_significance".\n'
        "Return ONLY the JSON."
    ),
}


def _make_context(node: dict, index_data: dict, max_n: int = 4) -> str:
    lines = []
    for cid in node.get("source_captions", [])[:max_n]:
        rec = index_data.get(cid)
        if rec and rec.get("caption"):
            lines.append(f"- {rec['caption'][:200]}")
    return "\n".join(lines) or "(no captions available)"


def call_llm_description(node: dict, index_data: dict, max_retries: int = 3) -> dict:
    ntype    = node["node_type"]
    template = LLM_PROMPTS.get(ntype, LLM_PROMPTS["condition"])
    prompt   = template.format(
        name    = node["canonical_name"],
        aliases = ", ".join(node.get("aliases", [])) or "N/A",
        context = _make_context(node, index_data),
    )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                timeout=60,
            )
            text = resp.choices[0].message.content.strip()
            #  markdown 
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$",          "", text)
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {"summary": text}
        except Exception as e:
            print(f"    [LLM  {attempt+1}/{max_retries}] {e}")
            time.sleep(2 * (attempt + 1))
    return {"summary": node["canonical_name"]}


def main():
    parser = argparse.ArgumentParser(description="MicroWorld S3：（NCBI  + LLM ）")
    parser.add_argument("--nodes",         default=str(NODES_PATH))
    parser.add_argument("--index",         default=str(SUPPORT_DIR / "dataset_index.json"))
    parser.add_argument("--output",        default=str(DESC_OUT_PATH))
    parser.add_argument("--resume",        action="store_true", help="")
    parser.add_argument("--node_types",    nargs="+",
                        default=["structure", "method", "condition", "property", "organism"],
                        help="")
    parser.add_argument("--ncbi_api_key",  default="",
                        help="NCBI API Key（， 10 req/s）\n"
                             "：https://www.ncbi.nlm.nih.gov/account/")
    parser.add_argument("--ncbi_only",     action="store_true",
                        help=" NCBI ， LLM（ NCBI  LLM）")
    parser.add_argument("--save_every",    type=int, default=20,
                        help="（ 20）")
    args = parser.parse_args()

    sleep = 0.11 if args.ncbi_api_key else 0.34
    print(f"[S3] NCBI ：{sleep:.2f}s / "
          + ("（ API Key）" if args.ncbi_api_key else "（ API Key，3 req/s ）"))

    with open(args.nodes, "r", encoding="utf-8") as f:
        nodes: list[dict] = json.load(f)
    with open(args.index, "r", encoding="utf-8") as f:
        index_data = {r["id"]: r for r in json.load(f)}

    descriptions: dict[str, dict] = {}
    if args.resume and Path(args.output).exists():
        with open(args.output, "r", encoding="utf-8") as f:
            descriptions = json.load(f)
        print(f"[S3] ： {len(descriptions)} 。")

    target = [n for n in nodes if n["node_type"] in args.node_types]
    print(f"[S3] ：{len(target)}（：{args.node_types}）")

    ncbi_hit = 0
    llm_hit  = 0

    for i, node in enumerate(target):
        eid  = node["entity_id"]
        name = node["canonical_name"]
        ntype = node["node_type"]

        if args.resume and eid in descriptions:
            continue

        print(f"[S3] {i+1}/{len(target)}  [{ntype}] {name}")

        # ── Step 1: NCBI ────────────────────────────────────────────────────
        ncbi_result = ncbi_lookup(node, args.ncbi_api_key, sleep)

        if ncbi_result:
            desc_dict = {**ncbi_result, "source": "ncbi"}
            ncbi_hit += 1
            print(f"       ✓ NCBI ({ncbi_result['ncbi_db']}): {ncbi_result['summary'][:80]}...")
        elif args.ncbi_only:
            desc_dict = {"summary": "", "source": "ncbi_miss", "ncbi_db": None, "ncbi_id": None}
            print(f"       ✗ NCBI miss（--ncbi_only  LLM）")
        else:
            # ── Step 2: LLM fallback ────────────────────────────────────────
            desc_dict = {**call_llm_description(node, index_data),
                         "source": "llm", "ncbi_db": None, "ncbi_id": None}
            llm_hit += 1
            print(f"       → LLM fallback: {desc_dict.get('summary','')[:80]}...")

        descriptions[eid] = {
            "entity_id":      eid,
            "canonical_name": name,
            "node_type":      ntype,
            **desc_dict,
        }

        if (i + 1) % args.save_every == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(descriptions, f, ensure_ascii=False, indent=2)
            print(f"  →  {len(descriptions)}  "
                  f"（NCBI {ncbi_hit} / LLM {llm_hit}）")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, ensure_ascii=False, indent=2)

    for node in nodes:
        eid = node["entity_id"]
        if eid in descriptions:
            node["description"] = descriptions[eid].get("summary", "")
    with open(args.nodes, "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)

    total = len(descriptions)
    print(f"\n[S3] 。 {total} 。")
    print(f"     NCBI ：{ncbi_hit} （{ncbi_hit/max(total,1)*100:.1f}%）")
    print(f"     LLM ： {llm_hit} （{llm_hit/max(total,1)*100:.1f}%）")
    print(f"     ：{args.output}")
    print(f"     nodes.json description ")
    print(f"\n[S3] 。：python microWorld_s4.py")


if __name__ == "__main__":
    main()

