import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

SUPPORT_DIR   = Path(__file__).parent / "support"
KG_DIR        = SUPPORT_DIR / "KG"
TRIPLETS_PATH = SUPPORT_DIR / "raw_triplets.json"
INDEX_PATH    = SUPPORT_DIR / "dataset_index.json"
NODES_PATH    = KG_DIR / "nodes.json"
EDGES_PATH    = KG_DIR / "edges.json"
GRAPH_PATH    = KG_DIR / "graph.json"

# node_type → node_type_id 
NODE_TYPE_ID: dict[str, int] = {
    'structure': 0,
    'method':    1,
    'condition': 2,
    'property':  3,
    'organism':  4,
    'image':     5,
}


def to_entity_id(name: str) -> str:
    """ canonical_name  entity_id→"""
    eid = name.strip().lower()
    eid = re.sub(r'[^\w\s-]', '', eid)
    eid = re.sub(r'\s+', '_', eid)
    eid = re.sub(r'_+', '_', eid).strip('_')
    return eid


def _classify_type_from_text(text: str, declared_type: str) -> str:
    """ structure """
    lower = text.lower()
    method_kws = {'microscopy', 'tomography', 'imaging', 'staining', 'spectroscopy',
                  'sequencing', 'electrophoresis', 'pcr', 'elisa', 'immunofluorescence',
                  'assay', 'western blot', 'flow cytometry', 'nanoscopy'}
    if any(k in lower for k in method_kws):
        return 'method'
    return declared_type


class EntityRegistry:
    """
    aliases 
    entity_id
    """

    def __init__(self):
        self._nodes: dict[str, dict] = {}        # entity_id → node dict
        self._alias_map: dict[str, str] = {}      # normalized_alias → entity_id

    def _normalize(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.strip().lower())

    def add_entity(self, canonical: str, entity_type: str,
                   aliases: list[str] = None,
                   mesh_id: str = None,
                   source_caption: str = None) -> str:
        #  canonical_name 
        norm_canonical = self._normalize(canonical)
        eid = (self._alias_map.get(norm_canonical)
               or to_entity_id(canonical))

        if eid not in self._nodes:
            # 
            self._nodes[eid] = {
                'node_type':     entity_type,
                'entity_id':     eid,
                'canonical_name': canonical.strip(),
                'mesh_id':       mesh_id,
                'aliases':       [],
                'description':   '',
                'source_captions': [],
                'degree':        0,
            }
            self._alias_map[norm_canonical] = eid
            self._alias_map[to_entity_id(canonical)] = eid
        else:
            #  mesh_id
            if mesh_id and not self._nodes[eid]['mesh_id']:
                self._nodes[eid]['mesh_id'] = mesh_id

        #  aliases
        if aliases:
            for alias in aliases:
                alias_norm = self._normalize(alias)
                if alias_norm != norm_canonical and alias_norm not in self._alias_map:
                    self._alias_map[alias_norm] = eid
                    if alias not in self._nodes[eid]['aliases']:
                        self._nodes[eid]['aliases'].append(alias)

        #  caption
        if source_caption and source_caption not in self._nodes[eid]['source_captions']:
            self._nodes[eid]['source_captions'].append(source_caption)

        return eid

    def resolve(self, name: str) -> str | None:
        """ entity_id None"""
        norm = self._normalize(name)
        eid  = self._alias_map.get(norm) or self._alias_map.get(to_entity_id(name))
        return eid

    def all_nodes(self) -> list[dict]:
        return list(self._nodes.values())

    def increment_degree(self, eid: str):
        if eid in self._nodes:
            self._nodes[eid]['degree'] += 1


def main():
    parser = argparse.ArgumentParser(description="MicroWorld S2KG ")
    parser.add_argument('--support_dir', default=None,
                        help='KG support  triplets/index/out_dir ')
    parser.add_argument('--triplets', default=str(TRIPLETS_PATH))
    parser.add_argument('--index',    default=str(INDEX_PATH))
    parser.add_argument('--out_dir',  default=str(KG_DIR))
    parser.add_argument('--min_degree', type=int, default=1,
                        help=' degree < min_degree image ')
    parser.add_argument('--no_cooccurs', action='store_true',
                        help=' co_occurs_with ')
    args = parser.parse_args()

    # --support_dir 
    if args.support_dir:
        sd = Path(args.support_dir)
        if args.triplets == str(TRIPLETS_PATH):
            args.triplets = str(sd / "raw_triplets.json")
        if args.index == str(INDEX_PATH):
            args.index = str(sd / "dataset_index.json")
        if args.out_dir == str(KG_DIR):
            args.out_dir = str(sd / "KG")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.triplets, encoding='utf-8') as f:
        triplets_data: list[dict] = json.load(f)
    with open(args.index, encoding='utf-8') as f:
        index_data: list[dict] = json.load(f)

    print(f"[S2]  {len(triplets_data)} {len(index_data)} ")

    img_path_map: dict[str, str] = {r['id']: r['image_path'] for r in index_data}

    registry = EntityRegistry()

    for rec in triplets_data:
        img_id = rec['id']
        for ent in rec.get('entities', []):
            canon = ent.get('canonical') or ent.get('text', '')
            if not canon:
                continue
            etype = _classify_type_from_text(canon, ent.get('type', 'structure'))
            registry.add_entity(
                canonical=canon,
                entity_type=etype,
                aliases=[ent.get('text', '')] if ent.get('text') != canon else [],
                mesh_id=ent.get('mesh_id'),
                source_caption=img_id,
            )

        # LLM  NER 
        for trip in rec.get('triplets', []):
            for role in ('head', 'tail'):
                name = trip.get(role, '').strip()
                if name:
                    rel = trip.get('relation', '')
                    #  entity_type
                    if role == 'tail' and rel == 'observed_by':
                        etype = 'method'
                    elif role == 'tail' and rel == 'has_property':
                        etype = 'property'
                    else:
                        etype = 'structure'
                    registry.add_entity(name, etype, source_caption=img_id)

    edges_raw: list[tuple[str, str, str]] = []   # (src_id, dst_id, relation)

    for rec in triplets_data:
        img_id = rec['id']

        for trip in rec.get('triplets', []):
            head_name = trip.get('head', '').strip()
            tail_name = trip.get('tail', '').strip()
            rel       = trip.get('relation', '')
            if not head_name or not tail_name or rel not in (
                'observed_by', 'has_property', 'located_in', 'interacts_with', 'part_of'
            ):
                continue
            src = registry.resolve(head_name)
            dst = registry.resolve(tail_name)
            if src and dst and src != dst:
                edges_raw.append((src, dst, rel))

        if not args.no_cooccurs:
            for pair in rec.get('co_occurs', []):
                if len(pair) != 2:
                    continue
                src = registry.resolve(pair[0])
                dst = registry.resolve(pair[1])
                if src and dst and src != dst:
                    edges_raw.append((src, dst, 'co_occurs_with'))

        for ent in rec.get('entities', []):
            canon = ent.get('canonical') or ent.get('text', '')
            eid   = registry.resolve(canon)
            if eid:
                edges_raw.append((img_id, eid, 'shown_in'))

    degree_counter: defaultdict[str, int] = defaultdict(int)
    for src, dst, _ in edges_raw:
        degree_counter[src] += 1
        degree_counter[dst] += 1

    all_entity_nodes = registry.all_nodes()
    kept_eids: set[str] = set()
    for node in all_entity_nodes:
        deg = degree_counter.get(node['entity_id'], 0)
        node['degree'] = deg
        if deg >= args.min_degree:
            kept_eids.add(node['entity_id'])

    img_nodes: list[dict] = []
    for i, rec in enumerate(index_data):
        img_id   = rec['id']
        img_path = img_path_map.get(img_id, '')
        img_nodes.append({
            'node_type':      'image',
            'entity_id':      img_id,
            'canonical_name': img_id,
            'mesh_id':        None,
            'aliases':        [img_path] if img_path else [],
            'description':    rec.get('caption', '')[:200],
            'source_captions': [img_id],
            'degree':         degree_counter.get(img_id, 0),
        })
        kept_eids.add(img_id)

    #  node_index 
    entity_nodes_kept = [n for n in all_entity_nodes if n['entity_id'] in kept_eids]
    all_nodes = entity_nodes_kept + img_nodes

    #  node_index  node_type_id
    for idx, node in enumerate(all_nodes):
        node['node_index']    = idx
        node['node_type_id']  = NODE_TYPE_ID.get(node['node_type'], 99)

    print(f"[S2] {len(all_nodes)}"
          f" {len(entity_nodes_kept)} {len(img_nodes)}")

    edge_set: set[tuple[str, str, str]] = set()
    edges_clean: list[tuple[str, str, str]] = []
    for src, dst, rel in edges_raw:
        if src not in kept_eids or dst not in kept_eids:
            continue
        key = (src, dst, rel)
        if key not in edge_set:
            edge_set.add(key)
            edges_clean.append((src, dst, rel))

    #  edges.json
    edges_list: list[dict] = []
    for rid, (src, dst, rel) in enumerate(edges_clean):
        edges_list.append({
            'src_id':       src,
            'dst_id':       dst,
            'relation_type': rel,
            'relation_id':  rid,
        })

    print(f"[S2] {len(edges_list)}")

    graph: dict[str, list[dict]] = defaultdict(list)
    for edge in edges_list:
        src = edge['src_id']
        dst = edge['dst_id']
        rel = edge['relation_type']
        graph[src].append({'neighbor': dst, 'relation': rel})
        #  shown_in  img → entity 
        if rel != 'shown_in':
            graph[dst].append({'neighbor': src, 'relation': rel})

    print(f"[S2] {len(graph)}")

    nodes_path = out_dir / 'nodes.json'
    edges_path = out_dir / 'edges.json'
    graph_path = out_dir / 'graph.json'

    with open(nodes_path, 'w', encoding='utf-8') as f:
        json.dump(all_nodes, f, ensure_ascii=False, indent=2)
    with open(edges_path, 'w', encoding='utf-8') as f:
        json.dump(edges_list, f, ensure_ascii=False, indent=2)
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(dict(graph), f, ensure_ascii=False, indent=2)

    rel_counter: defaultdict[str, int] = defaultdict(int)
    for e in edges_list:
        rel_counter[e['relation_type']] += 1

    print(f"\n[S2] ")
    print(f"     nodes.json : {len(all_nodes)}  → {nodes_path}")
    print(f"     edges.json : {len(edges_list)}   → {edges_path}")
    print(f"     graph.json : {graph_path}")
    print(f"     ")
    for rel, cnt in sorted(rel_counter.items()):
        print(f"       {rel:25s} : {cnt:6d}")
    print(f"\n[S2] python microWorld_s3.py --resume")


if __name__ == '__main__':
    main()
