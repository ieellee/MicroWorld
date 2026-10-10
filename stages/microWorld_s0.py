import argparse
import csv
import json
import os
import re
from pathlib import Path

SUPPORT_DIR = Path(__file__).parent / "support"
OUT_PATH    = SUPPORT_DIR / "dataset_index.json"

_BIO_KEYWORDS = [
    'microscop', 'confocal', 'fluorescen', 'immunostain', 'immunofluor',
    ' sem ', 'sem,', 'sem.', ' tem ', 'tem,', 'tem.', 'cryo-em', 'cryo em',
    'super-resolution', 'storm', ' palm ', 'sted ', 'fish ', 'fret',
    'western blot', 'flow cytometry', 'immunohistochem', ' ihc ', 'live-cell',
    'live cell', 'brightfield', 'phase contrast', 'dark field', 'tirf',
    'nucleus', 'nuclei', 'nucleolar', 'mitochondri', 'endoplasmic reticulum',
    ' golgi', 'lysosom', 'membrane', 'cytoplasm', 'cytoskelet', 'chromosome',
    'centrosom', 'centriole', 'ribosom', 'vesicle', 'focal adhesion',
    'neuron', 'astrocyte', 'fibroblast', 'macrophage', 'epithelial', 'stem cell',
    'cardiomyocyte', 'hepatocyte', 'lymphocyte', 'osteoblast', 'dendritic cell',
    'protein', 'gene ', 'genes ', ' dna ', ' rna ', ' mrna', 'antibody',
    'antigen', 'receptor', 'kinase', 'phosphorylat', 'ubiquitin', 'proteasom',
    'chromatin', 'histone', 'actin', 'tubulin', 'lamin ', 'collagen',
    'cancer', 'tumor', 'tumour', 'metastas', 'apoptosis', 'autophagy',
    'mitophagy', 'ferroptosis', 'senescen', 'inflammation', 'neurodegenerat',
    'fibrosis', 'infection', 'bacteria', 'pathogen', 'virus',
    'e. coli', 'e.coli', 'yeast', 'drosophila', 'zebrafish', 'c. elegans',
    'arabidopsis', 'hela', 'hek293', 'mcf-7', 'a549', 'u2os', 'cos-7',
    'transfect', 'knockdown', 'knockout', 'overexpression', 'crispr',
    'shrna', 'sirna', 'mirna', ' pcr', 'sequencing', 'immunoprecipitation',
    'dapi', 'gfp', 'yfp', 'rfp', 'mcherry', 'phalloidin', 'filipin',
    'lysotracker', 'mitotracker', 'annexin', 'propidium iodide', 'hoechst',
]

def is_biology_relevant(caption: str) -> bool:
    low = caption.lower()
    return any(kw in low for kw in _BIO_KEYWORDS)


def is_microvqa_csv(path: str) -> bool:
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
        return 'question_3' in cols and 'image_path' in cols
    except Exception:
        return False


def is_messages_format(path: str) -> bool:
    try:
        with open(path, encoding='utf-8') as f:
            first_line = f.readline().strip()
        if not first_line:
            return False
        obj = json.loads(first_line)
        return 'messages' in obj
    except Exception:
        return False


def parse_choices(choices_str: str) -> list[str]:
    parts = re.split(r'\d+\)\s*', choices_str)
    return [p.rstrip('; ').strip() for p in parts if p.strip()]


def build_microvqa_caption(row: dict, mode: str) -> str:
    question = row.get('question_3', '').strip()
    if mode == 'question':
        return question

    choices_str = row.get('choices_3', '')
    choices = parse_choices(choices_str)
    correct_idx = row.get('correct_index_3', '')

    if mode == 'answer':
        try:
            idx = int(correct_idx) - 1
            correct = choices[idx] if 0 <= idx < len(choices) else correct_idx
        except (ValueError, IndexError):
            correct = correct_idx
        return f"{question} Answer: {correct}"

    choices_text = '; '.join(f"{i+1}) {c}" for i, c in enumerate(choices))
    return f"{question}\nChoices:\n{choices_text}"


def remap_image_path(path: str, remap_pairs: list[tuple[str, str]]) -> str:
    for old, new in remap_pairs:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def normalize_record(rec: dict) -> dict:
    return {
        'id':         rec.get('id', ''),
        'image_path': rec.get('image_path', ''),
        'caption':    rec.get('caption', ''),
        'valid':      rec.get('valid', True),
        'meta':       rec.get('meta', {}),
    }


def load_microvqa_csv(path: str, caption_mode: str,
                      remap_pairs: list[tuple[str, str]]) -> list[dict]:
    records = []
    with open(path, newline='', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f)):
            img_path = remap_image_path(row.get('image_path', ''), remap_pairs)
            caption  = build_microvqa_caption(row, caption_mode)
            records.append({
                'id':         f'img_{i:05d}',
                'image_path': img_path,
                'caption':    caption,
                'valid':      bool(img_path and caption),
                'meta': {
                    '_question':     row.get('question_3', ''),
                    '_choices':      row.get('choices_3', ''),
                    '_correct_index': row.get('correct_index_3', ''),
                    '_task':         row.get('task_str', ''),
                },
            })
    return records


def parse_messages_format(obj: dict, remap_pairs: list[tuple[str, str]]) -> dict | None:
    messages = obj.get('messages', [])
    images   = obj.get('images', [])

    caption = ''
    for msg in messages:
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            if isinstance(content, str):
                caption = content.strip()
                break
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        caption = part.get('text', '').strip()
                        break

    img_path = ''
    if images:
        raw_path = images[0] if isinstance(images[0], str) else images[0].get('path', '')
        img_path = remap_image_path(raw_path, remap_pairs)

    if not img_path or not caption:
        return None

    return {
        'image_path': img_path,
        'caption':    caption,
        'valid':      True,
        'meta':       {'source': 'messages_format'},
    }


def load_jsonl(path: str, remap_pairs: list[tuple[str, str]]) -> list[dict]:
    records = []
    if is_messages_format(path):
        with open(path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rec = parse_messages_format(obj, remap_pairs)
                    if rec:
                        rec['id'] = f'img_{i:05d}'
                        records.append(rec)
                except json.JSONDecodeError:
                    pass
    else:
        with open(path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    img = remap_image_path(obj.get('image_path', ''), remap_pairs)
                    records.append({
                        'id':         obj.get('id', f'img_{i:05d}'),
                        'image_path': img,
                        'caption':    obj.get('caption', obj.get('text', '')),
                        'valid':      bool(img),
                        'meta':       obj.get('meta', {}),
                    })
                except json.JSONDecodeError:
                    pass
    return records


def load_json(path: str, remap_pairs: list[tuple[str, str]]) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = list(data.values())
    else:
        return []

    records = []
    for i, obj in enumerate(items):
        img = remap_image_path(obj.get('image_path', ''), remap_pairs)
        records.append({
            'id':         obj.get('id', f'img_{i:05d}'),
            'image_path': img,
            'caption':    obj.get('caption', obj.get('text', '')),
            'valid':      bool(img),
            'meta':       obj.get('meta', {}),
        })
    return records


def load_generic_csv(path: str, remap_pairs: list[tuple[str, str]]) -> list[dict]:
    records = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            img_col = next((c for c in reader.fieldnames
                            if 'image' in c.lower() and 'path' in c.lower()), None)
            cap_col = next((c for c in reader.fieldnames
                            if c.lower() in ('caption', 'text', 'description')), None)
            img = remap_image_path(row.get(img_col, ''), remap_pairs) if img_col else ''
            cap = row.get(cap_col, '') if cap_col else ''
            records.append({
                'id':         f'img_{i:05d}',
                'image_path': img,
                'caption':    cap,
                'valid':      bool(img and cap),
                'meta':       dict(row),
            })
    return records


def load_raw(path: str, caption_mode: str,
             remap_pairs: list[tuple[str, str]]) -> list[dict]:
    suffix = Path(path).suffix.lower()

    if suffix == '.csv':
        if is_microvqa_csv(path):
            return load_microvqa_csv(path, caption_mode, remap_pairs)
        return load_generic_csv(path, remap_pairs)

    if suffix in ('.jsonl', '.ndjson'):
        return load_jsonl(path, remap_pairs)

    if suffix == '.json':
        return load_json(path, remap_pairs)

    return load_jsonl(path, remap_pairs)


def main():
    parser = argparse.ArgumentParser(
        description="MicroWorld S0: Data preparation and indexing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input', required=True,
                        help='Input file path (CSV / JSON / JSONL)')
    parser.add_argument('--support_dir', default=None,
                        help='KG support directory (output defaults to support_dir/dataset_index.json)')
    parser.add_argument('--output', default=None,
                        help=f'Output path (default: {OUT_PATH} or support_dir/dataset_index.json)')
    parser.add_argument('--caption_mode', default='question',
                        choices=['question', 'answer', 'full'],
                        help='Caption construction mode (only applies to MicroVQA CSV)')
    parser.add_argument('--image_remap', nargs=2, metavar=('OLD', 'NEW'),
                        action='append', default=[],
                        help='Image path prefix replacement, can be specified multiple times')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for debugging)')
    parser.add_argument('--skip_invalid', action='store_true',
                        help='Skip records where valid=False')
    parser.add_argument('--filter_bio', action='store_true',
                        help='Keep only captions containing biological/microscopy keywords')
    args = parser.parse_args()

    if args.support_dir:
        support_dir = Path(args.support_dir)
        out_path = Path(args.output) if args.output else support_dir / "dataset_index.json"
    else:
        support_dir = SUPPORT_DIR
        out_path = Path(args.output) if args.output else OUT_PATH

    remap_pairs = [(old, new) for old, new in args.image_remap]

    print(f"[S0] Loading input: {args.input}")
    records = load_raw(args.input, args.caption_mode, remap_pairs)
    print(f"[S0] Raw records: {len(records)}")

    if args.filter_bio:
        before = len(records)
        records = [r for r in records if is_biology_relevant(r.get('caption', ''))]
        print(f"[S0] Bio filter: removed {before - len(records)} -> {len(records)} remaining")

    if args.skip_invalid:
        before = len(records)
        records = [r for r in records if r.get('valid')]
        print(f"[S0] Skip invalid: removed {before - len(records)} -> {len(records)} remaining")

    if args.max_samples is not None:
        records = records[:args.max_samples]
        print(f"[S0] Sample limit: {len(records)}")

    seen_ids: set[str] = set()
    for rec in records:
        original_id = rec['id']
        uid = original_id
        suffix_n = 0
        while uid in seen_ids:
            suffix_n += 1
            uid = f"{original_id}_{suffix_n}"
        rec['id'] = uid
        seen_ids.add(uid)

    support_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    valid_count = sum(1 for r in records if r.get('valid'))
    print(f"\n[S0] Done.")
    print(f"     Total records: {len(records)} (valid: {valid_count})")
    print(f"     Caption mode: {args.caption_mode}")
    print(f"     Bio filter: {'enabled' if args.filter_bio else 'disabled'}")
    print(f"     Output: {out_path}")
    next_script = "microWorld_s1_batch.py" if args.filter_bio else "microWorld_s1.py"
    print(f"\n[S0] Next step: python {next_script} --support_dir {support_dir}")


if __name__ == '__main__':
    main()
