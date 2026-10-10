import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

SUPPORT_DIR  = Path(__file__).parent / "support"
KG_DIR       = SUPPORT_DIR / "KG"
VIS_DIR      = SUPPORT_DIR / "visual_embeddings"
GRAPH_PATH   = KG_DIR / "graph.json"
NODES_PATH   = KG_DIR / "nodes.json"
ENT_SIM_PATH = SUPPORT_DIR / "entity_similarity_sorted.json"
IMG_SIM_PATH = SUPPORT_DIR / "image_similarity_sorted.json"


def build_neighbor_sets(graph: dict, entity_ids: set[str]) -> dict[str, set[str]]:
    """
    （ image  shown_in ）。
    """
    nbr_sets: dict[str, set[str]] = {}
    for eid in entity_ids:
        neighbors = graph.get(eid, [])
        nbr_sets[eid] = {
            n['neighbor'] for n in neighbors
            if not n['neighbor'].startswith('img_')
            and n.get('relation') != 'shown_in'
        }
    return nbr_sets


def compute_jaccard_row(eid: str,
                        nbr_set: set[str],
                        all_nbr_sets: dict[str, set[str]],
                        top_k: int) -> list[dict]:

    results = []
    if not nbr_set:
        return results
    for other_id, other_set in all_nbr_sets.items():
        if other_id == eid or not other_set:
            continue
        inter = len(nbr_set & other_set)
        if inter == 0:
            continue
        union = len(nbr_set | other_set)
        j = inter / union if union > 0 else 0.0
        if j > 0:
            results.append({'id': other_id, 'jaccard': round(j, 4)})
    results.sort(key=lambda x: x['jaccard'], reverse=True)
    return results[:top_k]



def _cosine_chunk_worker(
    i: int,
    embs: torch.Tensor,
    ids: list[str],
    top_k: int,
    chunk_size: int,
    n: int,
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    chunk_i = embs[i : i + chunk_size]
    sim_full = chunk_i @ embs.T
    hi = min(i + chunk_size, n)
    for local_r, global_r in enumerate(range(i, hi)):
        row = sim_full[local_r]
        row[global_r] = -2.0
        top_indices = row.topk(min(top_k, n - 1)).indices.tolist()
        eid = ids[global_r]
        out[eid] = [
            {'id': ids[j], 'cosine': round(row[j].item(), 4)}
            for j in top_indices
        ]
    return out


def compute_cosine_similarities(
    embs: torch.Tensor,
    ids: list[str],
    top_k: int,
    chunk_size: int = 512,
    workers: int = 1,
) -> dict[str, list[dict]]:

    n = len(ids)
    result: dict[str, list[dict]] = {id_: [] for id_ in ids}
    ranges = list(range(0, n, chunk_size))

    if workers <= 1 or len(ranges) <= 1:
        for idx, i in enumerate(ranges):
            part = _cosine_chunk_worker(i, embs, ids, top_k, chunk_size, n)
            result.update(part)
            print(f"  cosine  {i+1}~{min(i+chunk_size, n)}/{n} ...", end='\r')
        print()
        return result

    done = 0
    with ThreadPoolExecutor(max_workers=min(workers, len(ranges))) as ex:
        futs = {
            ex.submit(_cosine_chunk_worker, i, embs, ids, top_k, chunk_size, n): i
            for i in ranges
        }
        for fut in as_completed(futs):
            result.update(fut.result())
            done += 1
            print(f"  cosine  {done}/{len(ranges)} ...", end='\r')
    print()
    return result



def merge_scores(jaccard_rows: list[dict],
                 cosine_map: dict[str, float],
                 alpha: float,
                 top_k: int) -> list[dict]:
    """
     Jaccard  Cosine ， Top-K 。
    score = alpha * jaccard + (1-alpha) * cosine
    """
    # 
    all_ids = set(r['id'] for r in jaccard_rows) | set(cosine_map.keys())
    jaccard_map = {r['id']: r['jaccard'] for r in jaccard_rows}

    merged = []
    for cid in all_ids:
        j = jaccard_map.get(cid, 0.0)
        c = cosine_map.get(cid, 0.0)
        s = alpha * j + (1 - alpha) * c
        if s > 0:
            merged.append({
                'id':      cid,
                'jaccard': round(j, 4),
                'cosine':  round(c, 4),
                'score':   round(s, 4),
            })

    merged.sort(key=lambda x: x['score'], reverse=True)
    for rank, item in enumerate(merged[:top_k], 1):
        item['rank'] = rank
    return merged[:top_k]



def main():
    parser = argparse.ArgumentParser(description="MicroWorld S6：")
    parser.add_argument('--support_dir', default=None,
                        help='KG support （）')
    parser.add_argument('--graph',      default=str(GRAPH_PATH))
    parser.add_argument('--nodes',      default=str(NODES_PATH))
    parser.add_argument('--vis_dir',    default=str(VIS_DIR))
    parser.add_argument('--ent_out',    default=str(ENT_SIM_PATH))
    parser.add_argument('--img_out',    default=str(IMG_SIM_PATH))
    parser.add_argument('--top_k',      type=int,   default=50,
                        help='/ Top-K （ 50）')
    parser.add_argument('--alpha',      type=float, default=0.5,
                        help='Jaccard （ 0.5， Cosine）')
    parser.add_argument('--no_cosine',  action='store_true',
                        help=' Cosine （ Jaccard，）')
    parser.add_argument('--no_images',  action='store_true',
                        help='')
    parser.add_argument('--workers', type=int, default=None,
                        help='： Jaccard+、cosine （ min(32, CPU*2)）')
    args = parser.parse_args()

    workers = args.workers
    if workers is None:
        workers = min(32, (os.cpu_count() or 8) * 2)
    workers = max(1, workers)

    if args.support_dir:
        sd = Path(args.support_dir)
        if args.graph   == str(GRAPH_PATH):    args.graph   = str(sd / "KG" / "graph.json")
        if args.nodes   == str(NODES_PATH):    args.nodes   = str(sd / "KG" / "nodes.json")
        if args.vis_dir == str(VIS_DIR):       args.vis_dir = str(sd / "visual_embeddings")
        if args.ent_out == str(ENT_SIM_PATH):  args.ent_out = str(sd / "entity_similarity_sorted.json")
        if args.img_out == str(IMG_SIM_PATH):  args.img_out = str(sd / "image_similarity_sorted.json")

    vis_dir = Path(args.vis_dir)

    print("[S6]  ...")
    with open(args.graph, encoding='utf-8') as f:
        graph: dict = json.load(f)
    with open(args.nodes, encoding='utf-8') as f:
        nodes: list[dict] = json.load(f)

    entity_nodes = {n['entity_id'] for n in nodes if n['node_type'] != 'image'}
    image_nodes  = {n['entity_id'] for n in nodes if n['node_type'] == 'image'}

    print(f"[S6] ：{len(entity_nodes)}，：{len(image_nodes)}")

    vis_index_path = vis_dir / 'index.json'
    entity_ids_ordered: list[str] = []
    image_ids_ordered:  list[str] = []
    if vis_index_path.exists():
        with open(vis_index_path, encoding='utf-8') as f:
            vis_idx = json.load(f)
        entity_ids_ordered = vis_idx.get('entity_ids', [])
        image_ids_ordered  = vis_idx.get('image_ids', [])

    print("\n[S6]  ...")

    # 1. Jaccard（）
    nbr_sets = build_neighbor_sets(graph, entity_nodes)
    print(f"[S6] Jaccard （{len(entity_nodes)} ）...")

    # 2. Cosine（）
    cosine_ent: dict[str, dict[str, float]] = {}
    if not args.no_cosine:
        ent_emb_path = vis_dir / 'entity_embs.pt'
        if ent_emb_path.exists() and entity_ids_ordered:
            print("[S6]  ...")
            entity_embs = torch.load(ent_emb_path, map_location='cpu')
            entity_embs = torch.nn.functional.normalize(entity_embs, dim=-1)
            cosine_rows = compute_cosine_similarities(
                entity_embs, entity_ids_ordered, args.top_k, workers=workers
            )
            for eid, rows in cosine_rows.items():
                cosine_ent[eid] = {r['id']: r['cosine'] for r in rows}
        else:
            print("[S6] ， Cosine。")

    def _fuse_one_entity(eid: str) -> tuple[str, list[dict]]:
        nbr_set = nbr_sets.get(eid, set())
        jaccard_rows = compute_jaccard_row(eid, nbr_set, nbr_sets, args.top_k * 2)
        cosine_map = cosine_ent.get(eid, {})

        if not args.no_cosine and cosine_map:
            merged = merge_scores(jaccard_rows, cosine_map, args.alpha, args.top_k)
        else:
            merged = []
            for r in jaccard_rows[:args.top_k]:
                r['score'] = r['jaccard']
                r['rank'] = len(merged) + 1
                merged.append(r)
        return eid, merged

    entity_sim: dict[str, list[dict]] = {}
    entity_list = list(entity_nodes)
    if workers <= 1 or len(entity_list) <= 100:
        for eid in entity_list:
            k, v = _fuse_one_entity(eid)
            entity_sim[k] = v
    else:
        done = 0
        total = len(entity_list)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_fuse_one_entity, eid) for eid in entity_list]
            for fut in as_completed(futs):
                eid, merged = fut.result()
                entity_sim[eid] = merged
                done += 1
                if done % 500 == 0 or done == total:
                    print(f"   {done}/{total} ...", end='\r')
        print()

    with open(args.ent_out, 'w', encoding='utf-8') as f:
        json.dump(entity_sim, f, ensure_ascii=False, indent=2)
    print(f"[S6] ：{args.ent_out}")

    if not args.no_images:
        img_emb_path = vis_dir / 'image_embs.pt'
        if img_emb_path.exists() and image_ids_ordered:
            print("\n[S6]  ...")
            image_embs = torch.load(img_emb_path, map_location='cpu')
            image_embs = torch.nn.functional.normalize(image_embs, dim=-1)
            cosine_img_rows = compute_cosine_similarities(
                image_embs, image_ids_ordered, args.top_k, workers=workers
            )
            image_sim: dict[str, list[dict]] = {}
            for img_id, rows in cosine_img_rows.items():
                ranked = [{'id': r['id'], 'cosine': r['cosine'], 'rank': rank}
                          for rank, r in enumerate(rows, 1)]
                image_sim[img_id] = ranked

            with open(args.img_out, 'w', encoding='utf-8') as f:
                json.dump(image_sim, f, ensure_ascii=False, indent=2)
            print(f"[S6] ：{args.img_out}")
        else:
            print("[S6] ，。")

    print(f"\n[S6] Pipeline ！MicroWorld KG 。")
    print(f"     ：{args.ent_out}")
    print(f"      microworld.py 。")


if __name__ == '__main__':
    main()
