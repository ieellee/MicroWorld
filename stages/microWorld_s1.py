import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

URL   = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
KEY   = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4")

client = OpenAI(base_url=URL, api_key=KEY)

SUPPORT_DIR     = Path(__file__).parent / "support"
INDEX_PATH      = SUPPORT_DIR / "dataset_index.json"
TRIPLETS_PATH   = SUPPORT_DIR / "raw_triplets.json"

ALLOWED_RELATIONS = {"observed_by", "has_property", "located_in",
                     "interacts_with", "part_of"}

_LOCAL_MESH_DIR = os.path.expanduser("~/.scispacy/datasets/2023-04-23/mesh")
_MESH_CONFIGURED = False


def _configure_mesh():
    global _MESH_CONFIGURED
    if _MESH_CONFIGURED:
        return
    try:
        from scispacy.candidate_generation import LinkerPaths, DEFAULT_PATHS, DEFAULT_KNOWLEDGE_BASES
        from scispacy.linking_utils import KnowledgeBase

        local_dir = _LOCAL_MESH_DIR
        if os.path.exists(local_dir):
            DEFAULT_PATHS["mesh"] = LinkerPaths(
                ann_index=os.path.join(local_dir, "nmslib_index.bin"),
                tfidf_vectorizer=os.path.join(local_dir, "tfidf_vectorizer.joblib"),
                tfidf_vectors=os.path.join(local_dir, "tfidf_vectors_sparse.npz"),
                concept_aliases_list=os.path.join(local_dir, "concept_aliases.json"),
            )
            local_jsonl = os.path.join(local_dir, "umls_mesh_2022.jsonl")

            class LocalMesh(KnowledgeBase):
                def __init__(self, file_path=local_jsonl):
                    super().__init__(file_path)

            DEFAULT_KNOWLEDGE_BASES["mesh"] = LocalMesh
    except Exception as e:
        print(f"[S1] MeSH local path config failed ({e}), using default path.")
    _MESH_CONFIGURED = True


def _load_nlp(use_linker: bool = True):
    _configure_mesh()
    import spacy
    for model_name in ["en_core_sci_lg", "en_core_sci_sm"]:
        try:
            nlp = spacy.load(model_name)
            if use_linker:
                try:
                    nlp.add_pipe("scispacy_linker",
                                 config={"resolve_abbreviations": True,
                                         "linker_name": "mesh"})
                except Exception as e:
                    print(f"[S1] scispaCy linker failed ({e}), using NER only.")
            print(f"[S1] Loaded NLP model: {model_name}")
            return nlp
        except OSError:
            continue
    print("[S1] No scispaCy model found, NER will return empty list.")
    return None


_METHOD_KEYWORDS = {'microscopy', 'tomography', 'imaging', 'staining', 'sequencing',
                    'spectroscopy', 'electrophoresis', 'western', 'pcr', 'elisa',
                    'flow cytometry', 'assay', 'immunofluorescence', 'nanoscopy'}
_PROPERTY_KEYWORDS = {'asymmetric', 'helical', 'spherical', 'elongated', 'filamentous',
                      'globular', 'crystalline', 'amorphous', 'tubular', 'vesicular',
                      'clustered', 'dense', 'diffuse', 'punctate', 'ring-shaped'}

def _classify_entity_type(text: str, spacy_label: str) -> str:
    lower = text.lower()
    if any(kw in lower for kw in _METHOD_KEYWORDS):
        return 'method'
    if any(kw in lower for kw in _PROPERTY_KEYWORDS):
        return 'property'
    if spacy_label in ('TAXON', 'ORGANISM', 'SPECIES'):
        return 'organism'
    if spacy_label in ('DISEASE', 'CONDITION', 'PHENOTYPE'):
        return 'condition'
    return 'structure'


def extract_entities_from_caption(nlp, caption: str) -> list[dict]:
    if nlp is None:
        return []

    doc = nlp(caption)
    seen: set[str] = set()
    entities: list[dict] = []

    for ent in doc.ents:
        text  = ent.text.strip()
        lower = text.lower()

        if len(text) < 3 or lower in seen:
            continue
        seen.add(lower)

        canonical = re.sub(r'\s*\([^)]{1,6}\)', '', text).strip()

        mesh_id = None
        if hasattr(ent._, 'kb_ents') and ent._.kb_ents:
            mesh_id = ent._.kb_ents[0][0]

        entity_type = _classify_entity_type(text, ent.label_)

        entities.append({
            'text':      text,
            'type':      entity_type,
            'canonical': canonical,
            'mesh_id':   mesh_id,
        })

    for chunk in doc.noun_chunks:
        text  = chunk.text.strip()
        lower = text.lower()
        if len(text) < 4 or lower in seen:
            continue
        if all(t.is_stop or t.is_punct for t in chunk):
            continue
        seen.add(lower)
        entity_type = _classify_entity_type(text, 'ENTITY')
        entities.append({
            'text':      text,
            'type':      entity_type,
            'canonical': text,
            'mesh_id':   None,
        })

    return entities


_SYSTEM_PROMPT = """You are a biomedical knowledge graph extraction assistant specializing in microscopy image captions.
Your task is to extract structured relation triplets from a given image caption.

## Allowed Relation Types (use ONLY these five):

1. **observed_by**: A biological structure/entity is observed or imaged using a specific technique or instrument.
   - Pattern: <structure> is observed/imaged/shown/captured by/using <method>
   - Example: ("virus particle", "observed_by", "cryo-EM")

2. **has_property**: A biological entity possesses a physical, morphological, or quantitative property.
   - Pattern: <entity> is/appears <adjective> / <entity> with <property>
   - Example: ("structure", "has_property", "asymmetrical")

3. **located_in**: A biological entity is spatially located inside or on another entity.
   - Pattern: <entity> in/within/on/at <location>
   - Example: ("protein", "located_in", "membrane")

4. **interacts_with**: Two biological entities have a functional, physical, or chemical interaction.
   - Pattern: <entity A> binds/neutralizes/inhibits/interacts with <entity B>
   - Example: ("antibody", "interacts_with", "virus")

5. **part_of**: One entity is a structural sub-component of another entity.
   - Pattern: <sub-structure> of <whole-structure>
   - Example: ("spike protein", "part_of", "virion")

## Extraction Rules:
- Extract ONLY triplets that clearly belong to one of the five relation types above.
- Both head and tail must be concrete nouns or noun phrases from the caption.
- Do NOT invent information not present in the caption.
- Normalize entity names to their canonical form.
- A single caption may yield zero, one, or multiple triplets.
- If no valid triplet can be extracted, return an empty list.

## Output Format:
Return ONLY a JSON object: {"triplets": [{"head": "...", "relation": "...", "tail": "..."}]}
No explanation, markdown, or text outside the JSON."""

_USER_TEMPLATE = "Caption: {caption}"


def extract_triplets_from_response(text: str) -> list[dict]:
    code_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    raw_json = code_block.group(1) if code_block else None
    if raw_json is None:
        brace = re.search(r'\{.*\}', text, re.DOTALL)
        raw_json = brace.group() if brace else None
    if raw_json is None:
        return []
    try:
        data = json.loads(raw_json)
        triplets = data.get('triplets', [])
        return [
            t for t in triplets
            if isinstance(t, dict)
            and all(k in t for k in ('head', 'relation', 'tail'))
            and t['relation'] in ALLOWED_RELATIONS
        ]
    except json.JSONDecodeError:
        return []


def call_llm_relation_extraction(caption: str, max_retries: int = 3) -> list[dict]:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": _USER_TEMPLATE.format(caption=caption)},
                ],
                temperature=0.1,
                timeout=30,
            )
            text = resp.choices[0].message.content.strip()
            return extract_triplets_from_response(text)
        except Exception as e:
            wait = 2 * (attempt + 1)
            print(f"  [LLM retry {attempt+1}/{max_retries}] {e}, waiting {wait}s")
            time.sleep(wait)
    return []


def compute_co_occurs(entities: list[dict]) -> list[list[str]]:
    names = [e['canonical'] for e in entities]
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append([names[i], names[j]])
    return pairs


def process_record(nlp, rec: dict, use_llm: bool) -> dict:
    caption  = rec.get('caption', '')
    entities = extract_entities_from_caption(nlp, caption)
    triplets = call_llm_relation_extraction(caption) if use_llm and caption else []
    co_occurs = compute_co_occurs(entities)
    return {
        'id':         rec['id'],
        'caption':    caption,
        'entities':   entities,
        'triplets':   triplets,
        'co_occurs':  co_occurs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="MicroWorld S1: Entity extraction + LLM relation extraction"
    )
    parser.add_argument('--index',       default=str(INDEX_PATH))
    parser.add_argument('--output',      default=str(TRIPLETS_PATH))
    parser.add_argument('--support_dir', default=None,
                        help='KG support directory (auto-derives index/output paths)')
    parser.add_argument('--resume',      action='store_true',
                        help='Skip already-processed records (resume from checkpoint)')
    parser.add_argument('--no_llm',      action='store_true',
                        help='Skip LLM relation extraction (NER only, faster)')
    parser.add_argument('--no_linker',   action='store_true',
                        help='Skip scispaCy EntityLinker (no MeSH linking)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max records to process (for debugging)')
    parser.add_argument('--workers',     type=int, default=4,
                        help='LLM parallel threads (default 4)')
    parser.add_argument('--save_every',  type=int, default=50,
                        help='Intermediate save interval (default 50)')
    args = parser.parse_args()

    if args.support_dir:
        sd = Path(args.support_dir)
        if args.index  == str(INDEX_PATH):    args.index  = str(sd / "dataset_index.json")
        if args.output == str(TRIPLETS_PATH): args.output = str(sd / "raw_triplets.json")

    with open(args.index, encoding='utf-8') as f:
        index_data: list[dict] = json.load(f)
    print(f"[S1] Total records: {len(index_data)}")

    if args.max_samples:
        index_data = index_data[:args.max_samples]

    results: list[dict] = []
    done_ids: set[str] = set()
    out_path = Path(args.output)

    if args.resume and out_path.exists():
        with open(out_path, encoding='utf-8') as f:
            results = json.load(f)
        done_ids = {r['id'] for r in results}
        print(f"[S1] Resume: {len(done_ids)} records already done.")

    pending = [r for r in index_data if r['id'] not in done_ids]
    print(f"[S1] Pending: {len(pending)} records")
    print(f"[S1] LLM relation extraction: {'disabled' if args.no_llm else 'enabled'} (model={MODEL})")

    nlp = _load_nlp(use_linker=not args.no_linker)

    SUPPORT_DIR.mkdir(parents=True, exist_ok=True)

    def process_with_nlp(rec):
        result = process_record(nlp, rec, use_llm=not args.no_llm)
        return result

    if args.no_llm or args.workers <= 1:
        for i, rec in enumerate(pending):
            r = process_with_nlp(rec)
            results.append(r)
            print(f"[S1] {len(done_ids)+i+1}/{len(index_data)}  "
                  f"{rec['id']}  entities={len(r['entities'])}  triplets={len(r['triplets'])}")
            if (i + 1) % args.save_every == 0:
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"  -> Saved {len(results)} records")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_with_nlp, rec): rec for rec in pending}
            done_count = len(done_ids)
            for fut in as_completed(futures):
                rec = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    print(f"  [ERROR] {rec['id']}: {e}")
                    r = {'id': rec['id'], 'caption': rec.get('caption', ''),
                         'entities': [], 'triplets': [], 'co_occurs': []}
                results.append(r)
                done_count += 1
                print(f"[S1] {done_count}/{len(index_data)}  "
                      f"{rec['id']}  entities={len(r['entities'])}  triplets={len(r['triplets'])}")
                if done_count % args.save_every == 0:
                    with open(out_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    print(f"  -> Saved {len(results)} records")

    id_order = {r['id']: i for i, r in enumerate(index_data)}
    results.sort(key=lambda r: id_order.get(r['id'], 99999))

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_ents   = sum(len(r['entities']) for r in results)
    total_trips  = sum(len(r['triplets']) for r in results)
    print(f"\n[S1] Done.")
    print(f"     Records processed: {len(results)}")
    print(f"     Total entities: {total_ents}  avg {total_ents/max(len(results),1):.1f}/record")
    print(f"     Total triplets: {total_trips}  avg {total_trips/max(len(results),1):.1f}/record")
    print(f"     Output: {out_path}")
    print(f"\n[S1] Next step: python microWorld_s2.py")


if __name__ == '__main__':
    main()
