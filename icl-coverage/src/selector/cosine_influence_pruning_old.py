from __future__ import annotations
import sys
sys.path.append('/nas02/Hadi/Incontenxt-influence/DataInf/src')
sys.path.insert(1, '/nas02/Hadi/Incontenxt-influence/icl-coverage/src')
import attr
import numpy as np
from typing import Any
from pydantic import BaseModel, Extra
from datasets import Dataset
from dataloader import create_dataloaders, load_noisy_dataset_by_task
from lora_model import LORAEngine
from influence import IFEngine


from langchain.prompts.example_selector.base import BaseExampleSelector
from prompts.base import ExampleTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from selector.base import CommonSelectorArgs, SelectorUtilsMixin
from selector.greedy import decomposed_coverage_greedy
from tools.track import track

def best_comb_emb_coverage_v1(k, target_emb, cand_embs):
    init_skip_idxs = set()
    diffs = np.abs(cand_embs - target_emb[None, :])
    curr_comb = []
    curr_diff = np.array([np.inf for _ in range(target_emb.shape[0])])
    curr_obj = np.inf
    stats = dict(n_reset=0)
    while len(curr_comb) < k:
        best_idx = None
        best_diff = curr_diff
        best_obj = np.inf
        for idx, diff in enumerate(diffs):
            if idx in skip_idxs:
                continue
            cand_diff = np.minimum(diff, curr_diff)
            cand_obj = np.sum(cand_diff)
            if np.allclose(cand_obj, curr_obj):
                skip_idxs.add(idx)
            elif best_idx is None or cand_obj < best_obj:
                best_idx, best_diff, best_obj = idx, cand_diff, cand_obj
        if best_idx is None:
            skip_idxs = init_skip_idxs | set(curr_comb)
            stats['n_reset'] += 1
        else:
            curr_comb.append(best_idx)
            skip_idxs.add(best_idx)
            curr_diff, curr_obj = best_diff, best_obj
    return curr_comb, stats | dict(score=curr_obj)

@attr.s(auto_attribs=True)
class CosineInfluencePruningCoverageSelectorArgs(CommonSelectorArgs):
    emb_lm: str = 'sentence-transformers/all-mpnet-base-v2'
    coverage: bool = False
    reorder: bool = False

    def get_name(self):
        name_parts = [self.emb_lm.split('/')[-1]]
        if self.coverage: name_parts.append('coverage')
        if self.reorder: name_parts.append(f'reorder')
        return '-'.join(name_parts)


class CosineInfluencePruningCoverageSelector(BaseExampleSelector, SelectorUtilsMixin, BaseModel):
    args: CosineInfluencePruningCoverageSelectorArgs
    example_template: ExampleTemplate
    demo_candidates: Dataset

    embedding: Embeddings = None
    cand_embs: np.ndarray = None

    query2idx: dict[str, int] = None
    shot_scores_l: np.ndarray | list[np.ndarray] | None = None
    shot_idxs_l: np.ndarray | list[np.ndarray] | None = None

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def add_example(self, example: dict[str, str]) -> Any:
        ...

    @staticmethod
    def get_independent_shot_idxs(args, query_emb, cand_embs, return_scores=False):
        cand_scores = np.einsum('d,cd->c', query_emb, cand_embs)
        shot_idxs = np.argsort(cand_scores)[-args.n_shots:]
        if return_scores:
            return shot_idxs, cand_scores[shot_idxs]
        else:
            return shot_idxs

    @staticmethod
    def get_covering_shot_idxs(
        args, query_emb, cand_embs, cand_lens=None, max_len=-1, return_scores=False
    ):
        n_shots = args.n_shots
        cand_dimscores = np.einsum('d,cd->cd', query_emb, cand_embs)
        shot_idxs, _ = decomposed_coverage_greedy(
            n_shots, cand_dimscores, cand_lens=cand_lens, max_len=max_len)
        shot_scores = cand_dimscores[shot_idxs].sum(axis=-1)
        if args.reorder:
            order = np.argsort(shot_scores)
            shot_idxs = np.array(shot_idxs)[order]
            shot_scores = shot_scores[order]
        if return_scores:
            return np.array(shot_idxs), shot_scores
        else:
            return np.array(shot_idxs)

    @classmethod
    def get_shot_idxs(
        cls, args, query_emb, cand_embs, cand_lens=None, max_len=-1, return_scores=False
    ):
        if args.coverage:
            return cls.get_covering_shot_idxs(
                args, query_emb, cand_embs,
                cand_lens=cand_lens, max_len=max_len, return_scores=return_scores)
        else:
            return cls.get_independent_shot_idxs(
                args, query_emb, cand_embs, return_scores=return_scores)

    def select_examples(self, input_variables: dict[str, str], return_scores=False) -> list[dict]:
        query = self.example_template.format(**input_variables, embedding=True)
        if query not in self.query2idx:
            query_emb = np.array(self.embedding.embed_query(query))
            shot_idxs = self.get_shot_idxs(
                self.args, query_emb, self.cand_embs, return_scores=return_scores)
            if return_scores:
                shot_idxs, shot_scores = shot_idxs
        else:
            shot_idxs = self.shot_idxs_l[self.query2idx[query]]
            shot_scores = self.shot_scores_l[self.query2idx[query]]
        if return_scores:
            return self.demo_candidates.select(shot_idxs), shot_scores
        else:
            return self.demo_candidates.select(shot_idxs)

    @classmethod
    def from_examples(
        cls,
        args: CosineInfluencePruningCoverageSelectorArgs,
        examples: list[dict],
        example_template: BasePromptTemplate,
        query_examples: list[dict],
        enc_len_fn: Any = None,
        max_len: int = -1,
        subtract_gen_len: bool = False,
        device: str = 'cpu',
        progress_bar: bool = True,
    ) -> CosineInfluencePruningCoverageSelector:
        
        model_name_or_path="roberta-large"
        task="mrpc"
        noise_ratio=0.2
        batch_size=32
        target_modules=["value"]
        device="cuda"
        num_epochs=10
        lr=3e-4
        
        # mrpc_02_noise, noise_added=load_noisy_dataset_by_task(task="mrpc", noise_ratio=0.2)
        
        # fine-tuning models
        dataloader_outputs = create_dataloaders(model_name_or_path=model_name_or_path,
                                                task=task,
                                                noise_ratio=noise_ratio,
                                                batch_size=batch_size)
        train_dataloader, eval_dataloader, noise_index, tokenized_datasets, collate_fn = dataloader_outputs

        lora_engine = LORAEngine(model_name_or_path=model_name_or_path,
                                    target_modules=target_modules,
                                    train_dataloader=train_dataloader,
                                    eval_dataloader=eval_dataloader,
                                    device=device,
                                    num_epochs=num_epochs,
                                    lr=lr,
                                    low_rank=8, 
                                    task=task)

        lora_engine.build_LORA_model()
        lora_engine.train_LORA_model()  
        
        tr_grad_dict, val_grad_dict = lora_engine.compute_gradient(tokenized_datasets, collate_fn)
        
        influence_engine = IFEngine()
        influence_engine.preprocess_gradients(tr_grad_dict, val_grad_dict, noise_index)
        influence_engine.compute_hvps(compute_accurate=False)
        influence_engine.compute_IF()
        
        # vals are identity , proposed , LiSSA 
        
        # for method in influence_engine.IF_dict:
        
        #abs_vals=[abs(x) for x in influence_engine.IF_dict['identity']]
        abs_vals=[x for x in influence_engine.IF_dict['LiSSA']]
        num_to_trim=int(len(abs_vals)*0.05)
        print(num_to_trim)
        # abs_vals=abs_vals[0:num_to_trim]
        ids_to_skip=np.argsort(abs_vals)[::-1][0:num_to_trim]
        print(np.sort(abs_vals)[0:num_to_trim])
        print(np.sort(abs_vals)[::-1][0:num_to_trim])

        examples = cls.drop_duplicates(examples, example_template)
        ex_to_string = lambda ex: example_template.format(**ex, embedding=True)
        cand_strings_old = [ex_to_string(ex) for ex in examples]
        query_strings = [ex_to_string(ex) for ex in (query_examples or [])]
        print("Candidate Strings: {}".format(len(cand_strings_old)))
        cand_strings=[]
        for idx, string in enumerate(cand_strings_old):
            if idx in ids_to_skip:
                print('yes')
                continue
            else:
                cand_strings.append(string)
        print("Candidate Strings: {}".format(len(cand_strings)))
        print("Query Strings: {}".format(len(query_strings)))
        query2idx = {query: i for i, query in enumerate(query_strings)}

        max_len = max_len if max_len > 0 else 1000000
        cand_lens = [enc_len_fn(example_template.format(**ex)) if enc_len_fn else 0
            for ex in examples]
        query_lens = [enc_len_fn(example_template.format(**ex, test=True)) if enc_len_fn else 0
                      for ex in query_examples]
        completed_query_lens = [enc_len_fn(example_template.format(**ex)) if enc_len_fn else 0
                      for ex in query_examples]

        embedding = HuggingFaceEmbeddings(model_name=args.emb_lm, device=device)
        cand_embs = np.array(embedding.embed_documents(cand_strings))
        query_embs = np.array(embedding.embed_documents(query_strings))

        n_queries = len(query_examples)
        query_iter = track(range(n_queries), description='Finding shots', total=n_queries) if progress_bar else range(n_queries)
        # shot_idxs_l = np.empty((n_queries, args.n_shots), dtype=np.int32)
        # shot_scores_l = np.empty((n_queries, args.n_shots), dtype=np.float32)
        shot_idxs_l, shot_scores_l = [], []
        for idx in query_iter:
            # if idx in ids_to_skip:
            #     continue
            if not subtract_gen_len:
                _max_len = max_len - query_lens[idx]
            else:
                _max_len = max_len - completed_query_lens[idx] - 4
            shot_idxs, shot_scores = cls.get_shot_idxs(
                args, query_embs[idx], cand_embs, cand_lens, _max_len, True)
            shot_idxs_l.append(shot_idxs)
            shot_scores_l.append(shot_scores)

        return cls(
            args=args,
            example_template=example_template,
            demo_candidates=examples,
            # embedding=embedding,
            # cand_embs=cand_embs,
            query2idx=query2idx,
            shot_scores_l=shot_scores_l,
            shot_idxs_l=shot_idxs_l,
        )

def test():
    import numpy as np
    from functools import partial
    from pathlib import Path
    from driver import get_dataset, get_templates
    from constants import Dataset as DS
    from tools.track import track

    # dataset, input_feature, test_split = DS.SMCALFLOW_CS, 'paraphrase', 'test'
    dataset, input_feature, train_split, test_split = DS.GEOQUERY, 'source', 'template_1_train', 'template_1_test'
    # dataset, input_feature, train_split, test_split = DS.OVERNIGHT, 'paraphrase', 'socialnetwork_template_0_train', 'socialnetwork_template_0_test'
    # dataset, input_feature, test_split = DS.BREAK, 'question_text', 'validation'
    ds = get_dataset(dataset, data_root=Path('../data'))
    candidates = ds[train_split].select(list(range(min(500, len(ds[train_split])))))
    templates = get_templates(dataset, input_feature=input_feature)
    example_template = templates['example_template']

    if True:
        args = CosineInfluencePruningCoverageSelectorArgs(coverage=False)
        cosine_selector = CosineInfluencePruningCoverageSelector.from_examples(
            args, candidates, example_template, ds[test_split])

        from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
        from langchain.vectorstores import FAISS
        sim_selector = SemanticSimilarityExampleSelector.from_examples(
            examples=cosine_selector.demo_candidates,
            embeddings=HuggingFaceEmbeddings(model_name=args.emb_lm),
            vectorstore_cls=FAISS,
            example_template=example_template,
            k=args.n_shots, device='cpu', metric='IP')

        for idx in range(len(ds[test_split])):
            assert cosine_selector.select_examples(ds[test_split][idx])['source'] == [ex['source'] for ex in sim_selector.select_examples(ds[test_split][idx])], idx