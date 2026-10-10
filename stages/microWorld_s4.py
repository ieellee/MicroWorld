import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

SUPPORT_DIR  = Path(__file__).parent / "support"
KG_DIR       = SUPPORT_DIR / "KG"
VIS_DIR      = SUPPORT_DIR / "visual_embeddings"
INDEX_PATH   = SUPPORT_DIR / "dataset_index.json"
NODES_PATH   = KG_DIR / "nodes.json"
EDGES_PATH   = KG_DIR / "edges.json"
GRAPH_PATH   = KG_DIR / "graph.json"

MODEL_PATHS = {
    '2B': str(Path(__file__).parent / "Qwen3-VL-Embedding" / "models" / "Qwen3-VL-Embedding-2B"),
    '8B': str(Path(__file__).parent / "Qwen3-VL-Embedding" / "models" / "Qwen3-VL-Embedding-8B"),
}

sys.path.insert(0, str(Path(__file__).parent / "Qwen3-VL-Embedding"))


def load_embedder(model_size: str, device: str):
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
    model_path = MODEL_PATHS[model_size]
    print(f"[S4] ：{model_path}  (device={device})")
    embedder = Qwen3VLEmbedder(
        model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    return embedder


@torch.no_grad()
def embed_batch(embedder, inputs: list[dict], device: str) -> torch.Tensor:
    embs = embedder.process(inputs)
    if isinstance(embs, torch.Tensor):
        embs = embs.to(device)
    else:
        embs = torch.tensor(embs, dtype=torch.float32, device=device)
    # L2 ， = 
    embs = torch.nn.functional.normalize(embs, dim=-1)
    return embs.cpu()


def embed_images(embedder, image_paths: list[str], captions: list[str],
                 batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:

    img_embs_all   = []
    mixed_embs_all = []
    n = len(image_paths)

    for i in range(0, n, batch_size):
        batch_paths   = image_paths[i: i + batch_size]
        batch_captions = captions[i: i + batch_size]
        b = len(batch_paths)

        print(f"   {i+1}~{i+b}/{n} ...", end='\r')

        img_inputs = [
            {"image": p,
             "instruction": "Represent this microscopy image for knowledge graph node retrieval"}
            for p in batch_paths
        ]
        mixed_inputs = [
            {"image": p,
             "text": f"Caption: {cap}",
             "instruction": "Represent this image-text pair for biomedical knowledge retrieval"}
            for p, cap in zip(batch_paths, batch_captions)
        ]

        img_embs_all.append(embed_batch(embedder, img_inputs, device))
        mixed_embs_all.append(embed_batch(embedder, mixed_inputs, device))

    print()
    return torch.cat(img_embs_all, dim=0), torch.cat(mixed_embs_all, dim=0)


def embed_entities(embedder, entity_names: list[str],
                   batch_size: int, device: str) -> torch.Tensor:
    """。"""
    all_embs = []
    n = len(entity_names)
    for i in range(0, n, batch_size):
        batch = entity_names[i: i + batch_size]
        b = len(batch)
        print(f"   {i+1}~{i+b}/{n} ...", end='\r')
        inputs = [
            {"text": name,
             "instruction": "Represent this biomedical entity name for visual alignment"}
            for name in batch
        ]
        all_embs.append(embed_batch(embedder, inputs, device))
    print()
    return torch.cat(all_embs, dim=0)


def compute_visually_similar_edges(
    image_embs: torch.Tensor,
    image_ids: list[str],
    threshold: float,
    chunk_size: int = 256,
) -> list[tuple[str, str]]:

    n = len(image_ids)
    pairs: list[tuple[str, str]] = []

    for i in range(0, n, chunk_size):
        chunk_i = image_embs[i: i + chunk_size]
        for j in range(i, n, chunk_size):
            chunk_j = image_embs[j: j + chunk_size]
            sim = chunk_i @ chunk_j.T      # (ci, cj)
            #  (row, col)
            rows, cols = (sim >= threshold).nonzero(as_tuple=True)
            for r, c in zip(rows.tolist(), cols.tolist()):
                global_i = i + r
                global_j = j + c
                if global_i < global_j:  # 
                    pairs.append((image_ids[global_i], image_ids[global_j]))

    print(f"[S4]  {len(pairs)}  visually_similar （threshold={threshold}）")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="MicroWorld S4")
    parser.add_argument('--model',         default='2B', choices=['2B', '8B'],
                        help='2B')
    parser.add_argument('--device',        default='cuda:0')
    parser.add_argument('--batch',         type=int, default=4,
                        help='')
    parser.add_argument('--sim_threshold', type=float, default=0.85,
                        help='')
    parser.add_argument('--skip_images',   action='store_true',
                        help='')
    parser.add_argument('--skip_entities', action='store_true',
                        help='')
    parser.add_argument('--skip_edges',    action='store_true',
                        help=' visually_similar ')
    parser.add_argument('--support_dir', default=None,
                        help='KG support ')
    parser.add_argument('--index',   default=str(INDEX_PATH))
    parser.add_argument('--nodes',   default=str(NODES_PATH))
    parser.add_argument('--edges',   default=str(EDGES_PATH))
    parser.add_argument('--vis_dir', default=str(VIS_DIR))
    args = parser.parse_args()

    if args.support_dir:
        sd = Path(args.support_dir)
        if args.index   == str(INDEX_PATH): args.index   = str(sd / "dataset_index.json")
        if args.nodes   == str(NODES_PATH): args.nodes   = str(sd / "KG" / "nodes.json")
        if args.edges   == str(EDGES_PATH): args.edges   = str(sd / "KG" / "edges.json")
        if args.vis_dir == str(VIS_DIR):    args.vis_dir = str(sd / "visual_embeddings")

    vis_dir = Path(args.vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ──  ─────────────────────────────────────────────────────────────
    with open(args.index, encoding='utf-8') as f:
        index_data: list[dict] = json.load(f)
    with open(args.nodes, encoding='utf-8') as f:
        nodes: list[dict] = json.load(f)

    image_ids    = [r['id']           for r in index_data if r.get('valid', True)]
    image_paths  = [r['image_path']   for r in index_data if r.get('valid', True)]
    captions     = [r.get('caption', '') for r in index_data if r.get('valid', True)]

    entity_nodes = [n for n in nodes if n['node_type'] != 'image']
    entity_ids   = [n['entity_id']     for n in entity_nodes]
    entity_names = [n['canonical_name'] for n in entity_nodes]

    print(f"[S4] ：{len(image_ids)}，：{len(entity_ids)}")

    # ──  ─────────────────────────────────────────────────────────
    embedder = load_embedder(args.model, args.device)
    # embedder = load_embedder(args.model, "cuda:7")
    dim = None

    # ──  ─────────────────────────────────────────────────────────────
    image_embs = None
    if not args.skip_images:
        print(f"[S4]  ...")
        image_embs, mixed_embs = embed_images(
            embedder, image_paths, captions, args.batch, args.device
        )
        dim = image_embs.shape[1]
        torch.save(image_embs,  vis_dir / 'image_embs.pt')
        torch.save(mixed_embs,  vis_dir / 'image_mixed_embs.pt')
        print(f"[S4] ，shape={image_embs.shape}")

    # ──  ─────────────────────────────────────────────────────────────
    if not args.skip_entities and entity_names:
        print(f"[S4]  ...")
        entity_embs = embed_entities(
            embedder, entity_names, args.batch, args.device
        )
        if dim is None:
            dim = entity_embs.shape[1]
        torch.save(entity_embs, vis_dir / 'entity_embs.pt')
        print(f"[S4] ，shape={entity_embs.shape}")

    # ──  ─────────────────────────────────────────────────────────────
    index_info = {
        'image_ids':  image_ids,
        'entity_ids': entity_ids,
        'dim':        dim,
        'model':      args.model,
    }
    with open(vis_dir / 'index.json', 'w', encoding='utf-8') as f:
        json.dump(index_info, f, ensure_ascii=False, indent=2)

    # ── visually_similar  ───────────────────────────────────────────────────
    if not args.skip_edges and image_embs is not None:
        print(f"[S4]  visually_similar （threshold={args.sim_threshold}）...")
        sim_pairs = compute_visually_similar_edges(
            image_embs, image_ids, args.sim_threshold
        )

        if sim_pairs:
            # 
            with open(args.edges, encoding='utf-8') as f:
                edges: list[dict] = json.load(f)

            # 
            existing = {(e['src_id'], e['dst_id'], e['relation_type'])
                        for e in edges}
            next_rid = max((e['relation_id'] for e in edges), default=-1) + 1
            new_edges_count = 0
            for img_i, img_j in sim_pairs:
                key = (img_i, img_j, 'visually_similar')
                if key not in existing:
                    existing.add(key)
                    edges.append({
                        'src_id':        img_i,
                        'dst_id':        img_j,
                        'relation_type': 'visually_similar',
                        'relation_id':   next_rid,
                    })
                    next_rid += 1
                    new_edges_count += 1

            with open(args.edges, 'w', encoding='utf-8') as f:
                json.dump(edges, f, ensure_ascii=False, indent=2)

            #  graph.json
            graph: dict = defaultdict(list)
            for edge in edges:
                s, d, r = edge['src_id'], edge['dst_id'], edge['relation_type']
                graph[s].append({'neighbor': d, 'relation': r})
                if r != 'shown_in':
                    graph[d].append({'neighbor': s, 'relation': r})
            with open(GRAPH_PATH, 'w', encoding='utf-8') as f:
                json.dump(dict(graph), f, ensure_ascii=False, indent=2)

            print(f"[S4]  {new_edges_count}  visually_similar ")

    print(f"\n[S4] 。")
    print(f"     visual_embeddings/ : {vis_dir}")
    print(f"\n[S4] ：python microWorld_s5.py")


if __name__ == '__main__':
    main()
