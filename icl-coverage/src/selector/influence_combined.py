from __future__ import annotations
import sys
sys.path.append('/nas02/Hadi/Incontenxt-influence/DataInf/src')
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
class InfluenceScoreSelectorArgs(CommonSelectorArgs):
    def get_name(self):
        return 'Computing Influence'


class InfluenceScoreSelector(BaseExampleSelector, SelectorUtilsMixin, BaseModel):
    args: InfluenceScoreSelectorArgs
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
        args: InfluenceScoreSelectorArgs,
        examples: list[dict],
        example_template: ExampleTemplate,
        query_examples: list[dict] = None,
        enc_len_fn: Any = None,
        max_len: int = -1,
        subtract_gen_len: bool = False,
        device: str = 'cpu',
        progress_bar: bool = True,
    ) -> InfluenceScoreSelector:
        
        base_path = "/nas02/Hadi/Incontenxt-influence/DataInf/llama-2-13b-chat-converted" 
        project_path ="/nas02/Hadi/Incontenxt-influence/DataInf" 
        lora_engine = LORAEngineGeneration(base_path=base_path, 
                                        project_path=project_path,
                                        dataset_name='math_with_reason')
        # try:
        #     with open('./results_save_influencemmlu.pkl', 'rb') as f:
        #         IF_dict=pkl.load(f)
        #     print('loaded pickle')
        #     print(IF_dict)
        #     sorted_influences=IF_dict['influence']['proposed'].apply(lambda x: x.argsort(), axis=1)
        #     print(sorted_influences)
        # except Exception as e:
        #     print(e)
        print('creating datasets')
        tokenized_datasets, collate_fn = lora_engine.create_tokenized_datasets()
        tr_grad_dict, val_grad_dict = lora_engine.compute_gradient(tokenized_datasets, collate_fn)
        
        with open(f"./training_grad_dict_combined.pkl",'wb') as file:
            pkl.dump(tr_grad_dict, file)
        with open(f"./val_grad_dict_combined.pkl",'wb') as file:
            pkl.dump(val_grad_dict, file)
        
        train_dataset = datasets.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/datasets/banking77-train-900.hf")
        all_classes=set(train_dataset['label_text'])
        class_id_mapping={}
        for k,v in enumerate(train_dataset['label_text']):  #change based on dataset
            class_id_mapping[k]=v
        
        print('computing influences')
        influence_engine = IFEngineGeneration()
        influence_engine.preprocess_gradients(tr_grad_dict, val_grad_dict)
        influence_engine.compute_hvps()
        influence_engine.compute_IF()
        print(influence_engine.IF_dict['proposed'].shape)
        influence_engine.save_result()
        sorted_influences=influence_engine.IF_dict['proposed'].apply(lambda x: x.argsort(), axis=1)
        
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
            classes_picked=[]
            ids_picked=[]
            ids=sorted_influences.iloc[idx][::-1]
            # print(ids)
            for id in ids:
                if len(ids_picked)==8:
                    break
                elif id not in ids_picked and class_id_mapping[id] not in classes_picked: #pick top most id and check its class
                    ids_picked.append(id)
                    classes_picked.append(class_id_mapping[id])
                elif set(classes_picked)==set(all_classes):
                    classes_picked=[]
                    if id not in ids_picked and class_id_mapping[id] not in classes_picked: #pick top most id and check its class
                        ids_picked.append(id)
                        classes_picked.append(class_id_mapping[id])
                else:
                    continue
                
            shot_idxs_l.append(ids_picked)
            for id in ids_picked:
                # try:
                shot_scores_l.append(influence_engine.IF_dict['proposed'][id])    # IF_dict['influence']['proposed'][id]
                # except:
                #     shot_scores_l.append(IF_dict['influence']['proposed'][id])
                    
            # print(influence_engine.IF_dict['proposed'][ids])

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
    
    args = InfluenceScoreSelectorArgs(selector_type=ES.INFLUENCE,n_shots=8)
    #print(args)
    bs_selector = InfluenceScoreSelector.from_examples(args, candidates, example_template, query_examples=test, device=0)
    #print(bs_selector.demo_candidates)
    #print(bs_selector.query2idx)
    print(bs_selector.shot_scores_l)
    print(bs_selector.shot_idxs_l)
        