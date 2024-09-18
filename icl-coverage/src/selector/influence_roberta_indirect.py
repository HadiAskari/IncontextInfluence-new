from __future__ import annotations
import sys
# sys.path.append('/nas02/Hadi/Incontenxt-influence/DataInf/src')
sys.path.insert(1, '/nas02/Hadi/Incontenxt-influence/icl-coverage/src')
from selector.lora_model import LORAEngineGeneration
from selector.influence_generation import IFEngineGeneration
import datasets

import attr
import torch
import numpy as np
from typing import Any
from collections import defaultdict
from pydantic import BaseModel, Extra
from more_itertools import chunked
from datasets import Dataset
from bert_score.utils import get_tokenizer, get_model, model2layers
from langchain.prompts.example_selector.base import BaseExampleSelector
from prompts.base import ExampleTemplate
from selector.base import CommonSelectorArgs, SelectorUtilsMixin
from selector.greedy import decomposed_coverage_greedy
from tools.track import track
from datasets import load_dataset, Dataset
from constants import ExSel as ES
from numpy import argsort
import pickle as pkl
import os

from selector.dataloader_indirect import create_dataloaders, load_noisy_dataset_by_task
from selector.lora_model_indirect import LORAEngine
from selector.influence_generation import IFEngine


# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
    return load_dataset('gsm8k', 'main')

def get_templates():
    from prompts import GSM8KExampleTemplate
    task_desc = 'Answer the following question through careful, concise step-by-step reasoning.'
    return dict(
        prefix_template= task_desc,
        example_template=GSM8KExampleTemplate())


@attr.s(auto_attribs=True)
class RobertaInfluenceScoreSelectorArgs(CommonSelectorArgs):
    def get_name(self):
        return 'Computing Influence'


class RobertaInfluenceScoreSelector(BaseExampleSelector, SelectorUtilsMixin, BaseModel):
    args: RobertaInfluenceScoreSelectorArgs
    example_template: ExampleTemplate
    demo_candidates: Dataset
    query2idx: dict[str, int] = None
    shot_scores_l: np.ndarray | list[np.ndarray] | None = None
    shot_idxs_l: np.ndarray | list[np.ndarray] | None = None
    
    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True
    
    
    def add_example(self, example: dict[str, str]) -> Any:
        ...
    
    def select_examples(self, input_variables: dict[str, str], return_scores=False) -> list[dict]:
        query = self.example_template.format(**input_variables, embedding=True)
        if query not in self.query2idx:
            query_emb = np.array(self.embedding.embed_query(query))
            shot_idxs = self.get_shot_idxs(
                self.args, query_emb, self.cand_embs, return_scores=return_scores)
            if return_scores:
                shot_idxs, shot_scores = shot_idxs

        shot_idxs = self.shot_idxs_l[self.query2idx[query]]
        shot_scores = self.shot_scores_l[self.query2idx[query]]
        if return_scores:
            return self.demo_candidates.select(shot_idxs), shot_scores
        else:
            return self.demo_candidates.select(shot_idxs)
    
    @classmethod
    def from_examples(
        cls,
        name,
        args: RobertaInfluenceScoreSelectorArgs,
        examples: list[dict],
        example_template: ExampleTemplate,
        query_examples: list[dict] = None,
        enc_len_fn: Any = None,
        max_len: int = -1,
        subtract_gen_len: bool = False,
        device: str = 'cpu',
        progress_bar: bool = True,
        
    ) -> RobertaInfluenceScoreSelector:
    
        
        
        model_name_or_path="roberta-large"
        task="mrpc"
        noise_ratio=0
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
                                                batch_size=batch_size,
                                                dataset_name=name)
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
        influence_engine.save_result()
        
        # sorted_influences=influence_engine.IF_dict['proposed'].argsort()
        
        sorted_influences=influence_engine.IF_dict['identity'].apply(lambda x: x.argsort(), axis=1)
        
        
        
        # print(len(examples))
        examples = cls.drop_duplicates(examples, example_template)
        # print(len(examples))
        ex_to_string = lambda ex: example_template.format(**ex, embedding=True)
        cand_strings = [ex_to_string(ex) for ex in examples]
        query_strings = [ex_to_string(ex) for ex in (query_examples or [])]
        query2idx = {query: i for i, query in enumerate(query_strings)}
        #print(query2idx)
        n_queries = len(query_examples)
        query_iter = track(range(n_queries), description='Finding shots', total=n_queries) if progress_bar else range(n_queries)
        
        shot_idxs_l, shot_scores_l = [], []
        
        # len(query_iter)
        
        for idx in query_iter:
            #print(len(sorted_influences.iloc[idx]))
            
            ids=sorted_influences.iloc[idx][::-1][0:5]
            print(ids)
            shot_idxs_l.append(ids)
            for id in ids:
                # try:
                shot_scores_l.append(influence_engine.IF_dict['identity'][id])    # IF_dict['influence']['proposed'][id]
                # except:
                #     shot_scores_l.append(IF_dict['influence']['proposed'][id])
    

        shot_idxs_l=np.array(shot_idxs_l)
        shot_scores_l=np.array(shot_scores_l)
        
        return cls(
            args=args,
            example_template=example_template,
            demo_candidates=examples,
            # parser=parser,
            query2idx=query2idx,
            shot_scores_l=shot_scores_l,
            shot_idxs_l=shot_idxs_l,
        )
if __name__=='__main__':
    import numpy as np
    from functools import partial
    from pathlib import Path
    from langchain.prompts import FewShotPromptTemplate2
    #from data_utils import get_dataset, get_templates
    from constants import max_new_tokens_d, context_length_limit, LLM, Dataset as D
    from tools.lm import get_enc_len_fn
    from tools.track import track
    from constants import Dataset as DS
    
    dataset, input_feature, train_split, test_split = DS.GSM8K, None, 'train', 'test'
    ds = get_dataset(dataset, data_root=Path('../data'))
    candidates = ds[train_split].select([*range(0,90,1)])
    test= ds[test_split].select([*range(0,10,1)])
    templates = get_templates()
    example_template = templates['example_template']
    #print(example_template.templates)
    
    args = RobertaInfluenceScoreSelectorArgs(selector_type=ES.ROBERTAINFLUENCE,n_shots=8)
    #print(args)
    bs_selector = RobertaInfluenceScoreSelector.from_examples(args, candidates, example_template, query_examples=test, device=0)
    #print(bs_selector.demo_candidates)
    #print(bs_selector.query2idx)
    print(bs_selector.shot_scores_l)
    print(bs_selector.shot_idxs_l)
        