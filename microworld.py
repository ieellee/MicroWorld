import difflib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

_REL_DISPLAY = {
    'located_in':      'located in',
    'observed_by':     'observed by',
    'co_occurs_with':  'co-occurs with',
    'has_property':    'has property',
    'interacts_with':  'interacts with',
    'part_of':         'part of',
    'shown_in':        'shown in',
}


def _rel_str(rel: str) -> str:
    return _REL_DISPLAY.get(rel, rel.replace('_', ' '))

_STOPWORDS = {
    'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
    'from', 'and', 'or', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'this', 'that', 'these', 'those', 'it', 'its',
    'which', 'what', 'how', 'when', 'where', 'why', 'who', 'can', 'into',
    'as', 'using', 'used', 'using', 'image', 'images', 'cell', 'cells',
    'shown', 'shows', 'indicate', 'indicates', 'observed', 'appear',
}

_SHORT_CHARS = 3   

def _normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r'[^\w\s-]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def _is_valid_term(term: str) -> bool:
    t = term.strip()
    if len(t) < _SHORT_CHARS:
        return False
    words = t.lower().split()
    return not all(w in _STOPWORDS for w in words)


def _extract_summary(desc_val) -> str:
    if isinstance(desc_val, str):
        return desc_val
    if isinstance(desc_val, dict):
        for key in ('description', 'definition', 'summary', 'text', 'about', 'content'):
            v = desc_val.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        parts = [str(v) for v in desc_val.values()
                 if isinstance(v, str) and len(v) > 20]
        return ' '.join(parts)
    return str(desc_val) if desc_val else ''


class LocalMesh:
    _NLP_MODELS = ['en_core_sci_lg', 'en_core_sci_sm', 'en_core_web_lg']

    def __init__(self):
        self._nlp = None
        self._load()

    def _load(self):
        import spacy
        for model in self._NLP_MODELS:
            try:
                self._nlp = spacy.load(model, disable=['parser', 'ner_linker'])
                break
            except OSError:
                continue
        if self._nlp is None:
            raise RuntimeError(
                "No scispaCy model，please install en_core_sci_sm or en_core_sci_lg"
            )

    def extract(self, text: str) -> list[str]:
        if self._nlp is None:
            return []
        doc = self._nlp(text)
        seen: set[str] = set()
        result: list[str] = []
        for ent in doc.ents:
            t = ent.text.strip()
            key = _normalize(t)
            if key not in seen and _is_valid_term(t):
                seen.add(key)
                result.append(t)
        return result


_DEFAULT_SUPPORT_DIR = '/path/to/support'
_DEFAULT_VISUAL_EMBED_MODEL = 'Qwen3/Qwen3-VL-Embedding-8B'
_VISUAL_QUERY_INSTRUCTION = 'Represent this microscopy image for knowledge graph node retrieval'


class MicroWorld:
    def __init__(
        self,
        support_dir: str = None,
        no_nlp: bool = False,
        max_text_entities: int = 6,
        max_method_entities: int = 1,
        max_visual_entities: int = 3,
        freq_skip_ratio: float = 0.08,
        freq_compact_ratio: float = 0.04,
        max_context_chars: int = 6000,
        max_neighbors: int = 4,
        max_2hop: int = 3,
        no_2hop: bool = False,
        definition_only: bool = False,
        no_visually_similar: bool = False,
        fuzzy_cutoff: float = 0.72,
        visual_top_k_images: int = 3,
        visual_sim_threshold: float = 0.5,
        visual_embedding_model: str = None,
        visual_embed_batch_size: int = 16,
    ):
        if support_dir is None:
            support_dir = _DEFAULT_SUPPORT_DIR

        self.support_dir        = support_dir
        self.no_nlp             = no_nlp
        self.max_text_entities  = max_text_entities
        self.max_method_entities = max_method_entities
        self.max_visual_entities = max_visual_entities
        self.freq_skip_ratio    = freq_skip_ratio
        self.freq_compact_ratio = freq_compact_ratio
        self.max_context_chars  = max_context_chars
        self.max_neighbors      = max_neighbors
        self.max_2hop           = max_2hop
        self.no_2hop            = no_2hop
        self.definition_only    = definition_only
        self.no_visually_similar = no_visually_similar
        self.fuzzy_cutoff       = fuzzy_cutoff
        self.visual_top_k_images = visual_top_k_images
        self.visual_sim_threshold = visual_sim_threshold
        self.visual_embedding_model = visual_embedding_model or _DEFAULT_VISUAL_EMBED_MODEL
        self.visual_embed_batch_size = visual_embed_batch_size

        self.alias_to_id:      dict[str, str]        = {}
        self.id_to_node:       dict[str, dict]       = {}
        self.graph:            dict[str, list]        = {}
        self.descriptions:     dict[str, dict]       = {}
        self.entity_frequency: dict[str, float]      = {}
        self.image_to_entities: dict[str, list[str]] = {}
        self.path_to_image_id:  dict[str, str]       = {}

        self._nlp: Optional[LocalMesh] = None

        self._vis_image_embs = None   
        self._vis_image_ids: list[str] = []
        self._vis_embedder = None      
        self._vis_nn_cache: dict[str, list[dict]] = {}  

        print("[MicroWorld] loading KG...")
        self._load_kg()
        if not no_nlp:
            self._load_nlp()
        print("[MicroWorld] initial down")


    def _load_kg(self):
        sup = Path(self.support_dir)
        kg  = sup / 'KG'

        # nodes.json → alias_to_id, id_to_node
        with open(kg / 'nodes.json') as f:
            nodes: list[dict] = json.load(f)
        for node in nodes:
            eid  = node['entity_id']
            name = node['canonical_name']
            self.id_to_node[eid] = node
            self.alias_to_id[_normalize(name)] = eid
            for alias in node.get('aliases', []):
                self.alias_to_id[_normalize(alias)] = eid

        print(f"     节点数：{len(self.id_to_node)}  别名索引：{len(self.alias_to_id)}")

        # graph.json → graph, image_to_entities, entity_frequency
        with open(kg / 'graph.json') as f:
            self.graph = json.load(f)

        if self.no_visually_similar:
            dropped = 0
            for node_id in self.graph:
                orig = self.graph[node_id]
                filtered = [n for n in orig if n['relation'] != 'visually_similar']
                dropped += len(orig) - len(filtered)
                self.graph[node_id] = filtered
            print(f"     [ablation] 移除 {dropped} 条 visually_similar 邻接项")

        img_count   = 0
        ent_counter = Counter()
        for node_id, neighbors in self.graph.items():
            if node_id.startswith('img_'):
                img_count += 1
                ents = [n['neighbor'] for n in neighbors
                        if n['relation'] == 'shown_in']
                self.image_to_entities[node_id] = ents
                for eid in ents:
                    ent_counter[eid] += 1

        total_imgs = max(img_count, 1)
        self.entity_frequency = {
            eid: cnt / total_imgs for eid, cnt in ent_counter.items()
        }
        print(f"     视觉索引：{img_count} 张图像已建立 shown_in 实体映射（余弦检索用）")
        print(f"     实体频率已计算（基于 {img_count} 张图，共 {len(ent_counter)} 个实体）")

        # entity_descriptions.json
        desc_path = sup / 'entity_descriptions.json'
        if desc_path.exists():
            with open(desc_path) as f:
                self.descriptions = json.load(f)

        # dataset_index.json → path_to_image_id
        idx_path = sup / 'dataset_index.json'
        if idx_path.exists():
            with open(idx_path) as f:
                dataset_index: list[dict] = json.load(f)
            for item in dataset_index:
                img_id    = item.get('id', '')
                img_path  = item.get('image_path', '')
                if img_id and img_path:
                    self.path_to_image_id[img_path] = img_id
                    # 兼容不同分辨率路径（替换路径中的分辨率目录）
                    for res in ('512', '1024', '2048'):
                        alt = img_path.replace('/1024/', f'/{res}/')
                        self.path_to_image_id[alt] = img_id

    def _load_nlp(self):
        try:
            self._nlp = LocalMesh()
        except Exception as e:
            print(f"[MicroWorld] scispaCy 加载失败：{e}，将退回别名匹配模式。")
            self._nlp = None


    def _fuzzy_match_id(self, term: str) -> Optional[str]:
        norm = _normalize(term)
        candidates = list(self.alias_to_id.keys())
        matches = difflib.get_close_matches(norm, candidates,
                                            n=1, cutoff=self.fuzzy_cutoff)
        if matches:
            return self.alias_to_id[matches[0]]
        return None

    def _match_entity_to_kg(self, surface: str) -> Optional[dict]:
        norm = _normalize(surface)
        eid = self.alias_to_id.get(norm)
        if eid is None:
            eid = self._fuzzy_match_id(surface)
        if eid is None:
            return None
        node = self.id_to_node.get(eid, {})
        return {
            'entity_id':    eid,
            'canonical_name': node.get('canonical_name', eid),
            'node_type':    node.get('node_type', 'unknown'),
        }


    def _load_visual_index(self) -> bool:
        if self._vis_image_embs is not None:
            return True
        import torch
        vis_dir = Path(self.support_dir) / 'visual_embeddings'
        embs_path  = vis_dir / 'image_embs.pt'
        index_path = vis_dir / 'index.json'
        if not (embs_path.exists() and index_path.exists()):
            print(f"[MicroWorld] 未找到视觉嵌入索引（{vis_dir}），视觉检索不可用。")
            return False
        with open(index_path) as f:
            index = json.load(f)
        self._vis_image_ids = list(index['image_ids'])
        embs = torch.load(embs_path, map_location='cpu').float()
        self._vis_image_embs = torch.nn.functional.normalize(embs, dim=-1)
        print(f"[MicroWorld] 视觉索引已加载：{len(self._vis_image_ids)} 张图像，"
              f"dim={self._vis_image_embs.shape[1]}（建库模型 {index.get('model', '?')}）")
        return True

    def _load_visual_embedder(self):
        if self._vis_embedder is not None:
            return self._vis_embedder
        import sys as _sys
        import torch
        repo = str(Path(self.visual_embedding_model).parent.parent)
        if repo not in _sys.path:
            _sys.path.insert(0, repo)
        from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
        print(f"[MicroWorld] 加载视觉嵌入模型：{self.visual_embedding_model}")
        self._vis_embedder = Qwen3VLEmbedder(
            model_name_or_path=self.visual_embedding_model,
            torch_dtype=torch.bfloat16,
        )
        return self._vis_embedder

    def unload_visual_embedder(self):
        if self._vis_embedder is None:
            return
        import gc
        import torch
        del self._vis_embedder
        self._vis_embedder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[MicroWorld] 视觉嵌入模型已卸载。")

    @staticmethod
    def _encode_query_images(embedder, image_paths: list[str]):
        import torch
        inputs = [
            {'image': p, 'instruction': _VISUAL_QUERY_INSTRUCTION}
            for p in image_paths
        ]
        embs = embedder.process(inputs)  # process 内部已做 L2 归一化
        return embs.detach().float().cpu()

    def _nn_entities_from_emb(self, query_emb) -> list[dict]:
        import torch
        sims = self._vis_image_embs @ query_emb  # (N,) 余弦相似度
        k = max(1, min(self.visual_top_k_images, sims.shape[0]))
        top_sims, top_idx = torch.topk(sims, k)

        ent_best_sim: dict[str, float] = {}
        for sim, idx in zip(top_sims.tolist(), top_idx.tolist()):
            if sim < self.visual_sim_threshold:
                continue
            img_id = self._vis_image_ids[idx]
            for eid in self.image_to_entities.get(img_id, []):
                if eid not in self.id_to_node:
                    continue
                freq = self.entity_frequency.get(eid, 0.0)
                if freq > self.freq_skip_ratio:
                    continue
                if sim > ent_best_sim.get(eid, -1.0):
                    ent_best_sim[eid] = sim

        ranked = sorted(
            ent_best_sim.items(),
            key=lambda kv: (-kv[1], kv[0] not in self.descriptions,
                            self.entity_frequency.get(kv[0], 0.0)),
        )
        result = []
        for eid, sim in ranked[:self.max_visual_entities]:
            node = self.id_to_node[eid]
            result.append({
                'surface':       node.get('canonical_name', eid),
                'entity_id':     eid,
                'canonical_name': node.get('canonical_name', eid),
                'node_type':     node.get('node_type', 'unknown'),
                'source':        'visual',
                'visual_sim':    round(sim, 4),
            })
        return result

    def precompute_visual_nn(self, image_paths: list[str]):
        todo = [p for p in image_paths
                if p and p not in self._vis_nn_cache]
        if not todo:
            return
        if not self._load_visual_index():
            return
        embedder = self._load_visual_embedder()
        bs = max(1, self.visual_embed_batch_size)
        for i in range(0, len(todo), bs):
            batch = todo[i: i + bs]
            print(f"[MicroWorld] 视觉检索编码 {i+1}~{i+len(batch)}/{len(todo)} ...", end='\r')
            embs = self._encode_query_images(embedder, batch)
            for path, emb in zip(batch, embs):
                self._vis_nn_cache[path] = self._nn_entities_from_emb(emb)
        print(f"\n[MicroWorld] 视觉最近邻预计算完成：{len(todo)} 张图像。")

    def _get_visual_entities(self, image_path: str) -> list[dict]:
        if image_path in self._vis_nn_cache:
            return self._vis_nn_cache[image_path]
        if not self._load_visual_index():
            return []
        embedder = self._load_visual_embedder()
        emb = self._encode_query_images(embedder, [image_path])[0]
        result = self._nn_entities_from_emb(emb)
        self._vis_nn_cache[image_path] = result
        return result


    def _get_desc(self, entity_id: str, compact: bool = False) -> str:
        desc = self.descriptions.get(entity_id)
        if desc is None:
            node = self.id_to_node.get(entity_id, {})
            return node.get('canonical_name', entity_id)
        text = _extract_summary(desc.get('summary', '') if isinstance(desc, dict) else desc)
        if not text:
            text = _extract_summary(desc)
        if compact and len(text) > 120:
            text = text[:120] + '...'
        return text

    def _get_neighbors(self, entity_id: str, exclude: set[str]) -> dict:
        neighbors_raw = self.graph.get(entity_id, [])
        result: dict[str, str] = {}
        for n in neighbors_raw:
            nid = n['neighbor']
            rel = n['relation']
            if nid.startswith('img_') or rel == 'shown_in':
                continue
            if nid in exclude:
                continue
            result[nid] = rel
            if len(result) >= self.max_neighbors:
                break
        return result


    def _format_entity_block(
        self,
        idx: int,
        eid: str,
        surface: str,
        node_type: str,
        already_shown: set[str],
        compact: bool = False,
    ) -> str:
        node = self.id_to_node.get(eid, {})
        canonical = node.get('canonical_name', eid)

        lines: list[str] = []
        header = f"### {idx}. **{canonical}** [{node_type}]"
        if surface.lower() != canonical.lower():
            header += f"\n  _(matched from: \"{surface}\")_"
        lines.append(header)

        desc = self._get_desc(eid, compact=compact)
        if desc:
            lines.append(f"  Definition: {desc}")

        if not self.definition_only:
            neighbors = self._get_neighbors(eid, already_shown | {eid})
            if neighbors:
                lines.append("  Direct relations:")
                for nid, rel in neighbors.items():
                    n_node = self.id_to_node.get(nid, {})
                    n_name = n_node.get('canonical_name', nid)
                    n_type = n_node.get('node_type', 'unknown')
                    n_desc = self._get_desc(nid, compact=True)
                    rel_str = _rel_str(rel)
                    lines.append(f"    \u2022 {n_name} [{n_type}]  (relation: {rel_str})")
                    if n_desc:
                        lines.append(f"      {n_desc[:160]}{'...' if len(n_desc)>160 else ''}")

                # 2-hop
                if not self.no_2hop and len(neighbors) < 3:
                    hop2_shown: list[str] = []
                    for nid in list(neighbors.keys()):
                        hop2 = self._get_neighbors(nid,
                                                   already_shown | {eid} | set(neighbors))
                        for h2id, _ in list(hop2.items())[:self.max_2hop]:
                            if h2id not in hop2_shown:
                                hop2_shown.append(h2id)
                        if len(hop2_shown) >= self.max_2hop:
                            break
                    if hop2_shown:
                        lines.append("  Extended context (2-hop):")
                        for h2id in hop2_shown[:self.max_2hop]:
                            h2_node = self.id_to_node.get(h2id, {})
                            h2_name = h2_node.get('canonical_name', h2id)
                            h2_type = h2_node.get('node_type', 'unknown')
                            # 找经由哪个1-hop节点
                            via = next(
                                (self.id_to_node.get(nid, {}).get('canonical_name', nid)
                                 for nid in neighbors
                                 if any(n['neighbor'] == h2id
                                        for n in self.graph.get(nid, []))),
                                '?'
                            )
                            lines.append(f"    \u25e6 {h2_name} [{h2_type}]  (via {via})")

        return '\n'.join(lines)


    def build_context_from_entities(self, entities: list[str]) -> str:
        matched: list[dict] = []
        seen_ids: set[str] = set()
        method_count = 0

        for surface in entities:
            if not _is_valid_term(surface):
                continue
            hit = self._match_entity_to_kg(surface)
            if hit is None:
                continue
            eid = hit['entity_id']
            if eid in seen_ids:
                continue
            freq = self.entity_frequency.get(eid, 0.0)
            if freq > self.freq_skip_ratio:
                continue
            if hit['node_type'] == 'method':
                if method_count >= self.max_method_entities:
                    continue
                method_count += 1

            seen_ids.add(eid)
            matched.append({
                'surface':       surface,
                'entity_id':     eid,
                'canonical_name': hit['canonical_name'],
                'node_type':     hit['node_type'],
                'source':        'text',
            })
            if len(matched) >= self.max_text_entities:
                break

        return self._build_context_text(matched)

    def _build_context_text(self, matched_entities: list[dict]) -> str:
        if not matched_entities:
            return ''

        sections: list[str] = ["## Relevant Biological Knowledge (from MicroWorld KG)\n"]
        already_shown: set[str] = {e['entity_id'] for e in matched_entities}

        for i, ent in enumerate(matched_entities, 1):
            eid       = ent['entity_id']
            surface   = ent['surface']
            node_type = ent['node_type']
            freq      = self.entity_frequency.get(eid, 0.0)
            compact   = freq > self.freq_compact_ratio

            block = self._format_entity_block(
                i, eid, surface, node_type, already_shown, compact=compact
            )
            sections.append(block + '\n')

        ctx = '\n'.join(sections)
        if len(ctx) > self.max_context_chars:
            ctx = ctx[:self.max_context_chars] + '\n...[truncated]'
        return ctx

    def build_prompt(self, question: str, image_path: str = None) -> dict:
        matched: list[dict] = []
        seen_ids: set[str]  = set()
        method_count = 0

        if not self.no_nlp:
            candidates: list[str] = []
            if self._nlp is not None:
                candidates = self._nlp.extract(question)
            else:
                words = re.findall(r"[A-Za-z][\w'-]*", question)
                for w in range(len(words)):
                    for gram_n in (3, 2, 1):
                        phrase = ' '.join(words[w: w + gram_n])
                        if _is_valid_term(phrase):
                            candidates.append(phrase)

            for surface in candidates:
                if not _is_valid_term(surface):
                    continue
                hit = self._match_entity_to_kg(surface)
                if hit is None:
                    continue
                eid = hit['entity_id']
                if eid in seen_ids:
                    continue
                freq = self.entity_frequency.get(eid, 0.0)
                if freq > self.freq_skip_ratio:
                    continue
                if hit['node_type'] == 'method':
                    if method_count >= self.max_method_entities:
                        continue
                    method_count += 1
                seen_ids.add(eid)
                matched.append({
                    'surface':       surface,
                    'entity_id':     eid,
                    'canonical_name': hit['canonical_name'],
                    'node_type':     hit['node_type'],
                    'source':        'text',
                })
                if len(matched) >= self.max_text_entities:
                    break

        if image_path and self.max_visual_entities > 0:
            vis_ents = self._get_visual_entities(image_path)
            for vent in vis_ents:
                if vent['entity_id'] not in seen_ids:
                    seen_ids.add(vent['entity_id'])
                    matched.append(vent)

        knowledge_context = self._build_context_text(matched)

        prompt = question
        if knowledge_context:
            prompt = knowledge_context + '\n\n' + question

        return {
            'prompt':            prompt,
            'image_path':        image_path,
            'matched_entities':  matched,
            'knowledge_context': knowledge_context,
        }

    def run(self, question: str, image_path: str = None) -> None:
        result = self.build_prompt(question, image_path=image_path)
        print("=== Matched Entities ===")
        for ent in result['matched_entities']:
            print(f"  [{ent['source']}] {ent['canonical_name']} ({ent['node_type']})")
        print("\n=== Knowledge Context ===")
        print(result['knowledge_context'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MicroWorld KG 检索调试工具')
    parser.add_argument('question', nargs='?', default='cryo-electron tomography neurons mitochondria')
    parser.add_argument('--image_path', default=None)
    parser.add_argument('--support_dir', default=None)
    parser.add_argument('--no_nlp', action='store_true')
    args = parser.parse_args()

    mw = MicroWorld(support_dir=args.support_dir, no_nlp=args.no_nlp)
    mw.run(args.question, image_path=args.image_path)
