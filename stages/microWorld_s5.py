import argparse
import json
from collections import defaultdict
from pathlib import Path

SUPPORT_DIR  = Path(__file__).parent / "support"
KG_DIR       = SUPPORT_DIR / "KG"
NODES_PATH   = KG_DIR / "nodes.json"
GRAPH_PATH   = KG_DIR / "graph.json"
OUT_PATH     = KG_DIR / "results_close_entity.json"
STAT_PATH    = KG_DIR / "neighbors_stat.txt"

# degree 
THRESH_LOW  = 3    # degree ≤ THRESH_LOW → 1-hop + 2-hop
THRESH_HIGH = 20   # degree > THRESH_HIGH → 1-hop  50


def compute_neighbors(graph: dict[str, list[dict]],
                      entity_id: str,
                      degree: int,
                      max_1hop_low: int,
                      max_1hop_mid: int,
                      max_1hop_high: int,
                      max_2hop: int,
                      ) -> tuple[list[str], list[str]]:
    """
     entity_id  1-hop  2-hop （ image ）。

     (direct_neighbors, hop2_neighbors)。
    """
    raw_neighbors = graph.get(entity_id, [])
    #  image （img_xxxxx） shown_in 
    direct = [
        n['neighbor'] for n in raw_neighbors
        if not n['neighbor'].startswith('img_')
        and n.get('relation') != 'shown_in'
    ]

    #  degree 
    if degree <= THRESH_LOW:
        direct = direct[:max_1hop_low]
    elif degree <= THRESH_HIGH:
        direct = direct[:max_1hop_mid]
    else:
        direct = direct[:max_1hop_high]

    # 2-hop（ degree ≤ THRESH_LOW ）
    hop2: list[str] = []
    if degree <= THRESH_LOW:
        direct_set = set(direct) | {entity_id}
        for nbr in direct:
            for hop2_nbr in graph.get(nbr, []):
                nid = hop2_nbr['neighbor']
                if (nid not in direct_set
                        and not nid.startswith('img_')
                        and hop2_nbr.get('relation') != 'shown_in'):
                    hop2.append(nid)
                    direct_set.add(nid)   # 
                if len(hop2) >= max_2hop:
                    break
            if len(hop2) >= max_2hop:
                break

    return direct, hop2


def main():
    parser = argparse.ArgumentParser(description="MicroWorld S5：K-hop ")
    parser.add_argument('--support_dir', default=None,
                        help='KG support （）')
    parser.add_argument('--nodes',       default=str(NODES_PATH))
    parser.add_argument('--graph',       default=str(GRAPH_PATH))
    parser.add_argument('--output',      default=str(OUT_PATH))
    parser.add_argument('--stat',        default=str(STAT_PATH))
    parser.add_argument('--max_1hop_low',  type=int, default=200,
                        help='degree ≤ 3  1-hop （ 200）')
    parser.add_argument('--max_1hop_mid',  type=int, default=100,
                        help='3 < degree ≤ 20  1-hop （ 100）')
    parser.add_argument('--max_1hop_high', type=int, default=50,
                        help='degree > 20  1-hop （ 50）')
    parser.add_argument('--max_2hop',      type=int, default=200,
                        help='2-hop （ 200）')
    args = parser.parse_args()

    if args.support_dir:
        sd = Path(args.support_dir)
        if args.nodes  == str(NODES_PATH): args.nodes  = str(sd / "KG" / "nodes.json")
        if args.graph  == str(GRAPH_PATH): args.graph  = str(sd / "KG" / "graph.json")
        if args.output == str(OUT_PATH):   args.output = str(sd / "KG" / "results_close_entity.json")
        if args.stat   == str(STAT_PATH):  args.stat   = str(sd / "KG" / "neighbors_stat.txt")

    print(f"[S5]  ...")
    with open(args.nodes, encoding='utf-8') as f:
        nodes: list[dict] = json.load(f)
    with open(args.graph, encoding='utf-8') as f:
        graph: dict = json.load(f)

    #  image 
    entity_nodes = [n for n in nodes if n['node_type'] != 'image']
    print(f"[S5] ：{len(entity_nodes)}")

    results: dict[str, dict] = {}
    degree_stats = defaultdict(int)

    for i, node in enumerate(entity_nodes):
        eid   = node['entity_id']
        deg   = node.get('degree', len(graph.get(eid, [])))

        direct, hop2 = compute_neighbors(
            graph, eid, deg,
            args.max_1hop_low, args.max_1hop_mid, args.max_1hop_high, args.max_2hop
        )

        results[eid] = {
            'entity_id':       eid,
            'node_type':       node['node_type'],
            'canonical_name':  node['canonical_name'],
            'degree':          deg,
            'direct_neighbors': direct,
            '2hop_neighbors':  hop2,
        }

        # 
        if deg <= THRESH_LOW:
            degree_stats['low (≤3)'] += 1
        elif deg <= THRESH_HIGH:
            degree_stats['mid (4-20)'] += 1
        else:
            degree_stats['high (>20)'] += 1

        if (i + 1) % 500 == 0:
            print(f"  [S5] {i+1}/{len(entity_nodes)} ...")

    # 
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 
    stat_lines = [
        "MicroWorld S5 K-hop ",
        "=" * 40,
        f"：{len(entity_nodes)}",
        "",
        "degree ：",
    ]
    for tier, cnt in degree_stats.items():
        stat_lines.append(f"  {tier:20s}：{cnt:6d}")

    avg_direct = sum(len(v['direct_neighbors']) for v in results.values()) / max(len(results), 1)
    avg_hop2   = sum(len(v['2hop_neighbors']) for v in results.values()) / max(len(results), 1)
    stat_lines += [
        "",
        f" 1-hop ：{avg_direct:.1f}",
        f" 2-hop ：{avg_hop2:.1f}",
        "",
        f"：{out_path}",
    ]
    stat_text = '\n'.join(stat_lines)
    with open(args.stat, 'w', encoding='utf-8') as f:
        f.write(stat_text)

    print(f"\n[S5] 。")
    print(f"     ：{len(results)}")
    print(f"      1-hop：{avg_direct:.1f}  2-hop：{avg_hop2:.1f}")
    print(f"     ：{out_path}")
    print(f"     ：{args.stat}")
    print(f"\n[S5] ：python microWorld_s6.py")


if __name__ == '__main__':
    main()
