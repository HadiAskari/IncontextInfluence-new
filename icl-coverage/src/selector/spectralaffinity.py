from __future__ import annotations
import sys
sys.path.append('/nas02/Hadi/Incontenxt-influence/DataInf/src')
sys.path.insert(1, '/nas02/Hadi/Incontenxt-influence/icl-coverage/src')
from selector.lora_model import LORAEngineGeneration
from selector.influence_generation import IFEngineGeneration
import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

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
from tqdm.auto import tqdm
from sklearn.random_projection import SparseRandomProjection


def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
    return load_dataset('gsm8k', 'main')

def get_templates():
    from prompts import GSM8KExampleTemplate
    task_desc = 'Answer the following question through careful, concise step-by-step reasoning.'
    return dict(
        prefix_template= task_desc,
        example_template=GSM8KExampleTemplate())

@attr.s(auto_attribs=True)
class SpectralAffinityScoreSelectorArgs(CommonSelectorArgs):
    def get_name(self):
        return 'Computing Kmeans-centroid'
    

class SpectralAffinityScoreSelector(BaseExampleSelector, SelectorUtilsMixin, BaseModel):
    args: SpectralAffinityScoreSelectorArgs
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
        args: SpectralAffinityScoreSelectorArgs,
        examples: list[dict],
        example_template: ExampleTemplate,
        query_examples: list[dict] = None,
        enc_len_fn: Any = None,
        max_len: int = -1,
        subtract_gen_len: bool = False,
        device: str = 'cpu',
        progress_bar: bool = True,
    ) -> SpectralAffinityScoreSelector:
        
        base_path = "/nas02/Hadi/Incontenxt-influence/DataInf/llama-2-13b-chat-converted" 
        project_path ="/nas02/Hadi/Incontenxt-influence/DataInf" 
        lora_engine = LORAEngineGeneration(base_path=base_path, 
                                        project_path=project_path,
                                        dataset_name=name)

        print('creating datasets')
        try:
            with open(f"./training_grad_dict.pkl",'rb') as file:
                tr_grad_dict=pkl.load(file)
            with open(f"./val_grad_dict.pkl",'rb') as file:
                val_grad_dict=pkl.load(file)
        except:    
            print("In Except")
            tokenized_datasets, collate_fn = lora_engine.create_tokenized_datasets()
            tr_grad_dict, val_grad_dict = lora_engine.compute_gradient(tokenized_datasets, collate_fn)
        
        train_tensor_list=[]

        for k,v in tr_grad_dict.items():
            #print(k)
            layer_tensors=[]
            count=0
            for kk, vv in v.items():
                count=+1
                #print(kk)
                #if 'embed_tokens' in kk: # or '.1.' in kk or '.2.' in kk:
                # if '77' in kk or '78' in kk or '79' in kk:
                # if '37' in kk or '38' in kk or '39' in kk:
                if '.0.' in kk or '.1.' in kk or '38' in kk or '39' in kk: # or '79' in kk:
                    if vv.ndim==1:
                        vv=vv.view(1, -1)
                    layer_tensors.append(vv)
            train_tensor_list.append(layer_tensors)

        padded_tensors = []
        for row in tqdm(train_tensor_list):
            padded_row = []
            for tensor in row:
                #print(tensor.shape)
                if tensor.shape[1]==8192 or tensor.shape[1]==5120:
                    padded_row.append(tensor.float())

                else:
                    #print(tensor.shape[1])          
                    padded_tensor = torch.cat([tensor, torch.zeros(8,8192 - tensor.shape[1], dtype=tensor.dtype)], dim=1)
                    padded_row.append(padded_tensor.float())
                    
            padded_tensors.append(torch.stack(padded_row))

        padded_matrix=np.array(padded_tensors)

        flattened=[]
        for i in range(len(padded_matrix)):
            flattened.append(padded_matrix[i].flatten())

        val_tensor_list=[]

        for k,v in val_grad_dict.items():
            #print(k)
            layer_tensors=[]
            count=0
            for kk, vv in v.items():
                count=+1
                #print(kk)
                #if 'embed_tokens' in kk: # or '.1.' in kk or '.2.' in kk:
                # if '77' in kk or '78' in kk or '79' in kk:
                # if '37' in kk or '38' in kk or '39' in kk:
                if '.0.' in kk or '.1.' in kk or '38' in kk or '39' in kk: # or '79' in kk:
                    if vv.ndim==1:
                        vv=vv.view(1, -1)
                    layer_tensors.append(vv)
            val_tensor_list.append(layer_tensors)

        padded_val_tensors = []
        for row in tqdm(val_tensor_list):
            padded_row = []
            for tensor in row:
                #print(tensor.shape)
                if tensor.shape[1]==8192 or tensor.shape[1]==5120:
                    padded_row.append(tensor.float())

                else:
                    #print(tensor.shape[1])          
                    padded_tensor = torch.cat([tensor, torch.zeros(8,8192 - tensor.shape[1], dtype=tensor.dtype)], dim=1)
                    padded_row.append(padded_tensor.float())
                    
            padded_val_tensors.append(torch.stack(padded_row))

        padded_val_matrix=np.array(padded_val_tensors)
        
        flattened_val=[]
        for i in range(len(padded_val_matrix)):
            flattened_val.append(padded_val_matrix[i].flatten())
            
        new_flat=flattened+flattened_val
        
        rng = np.random.RandomState(42)
        transformer = SparseRandomProjection(random_state=rng)
        Combined = transformer.fit_transform(new_flat)
        
        X_new=Combined[0:84]
        Y_new=Combined[84:]
        
        clustering = SpectralClustering(n_clusters=18,
            assign_labels='discretize',
            affinity='rbf',
        random_state=0).fit(Combined)

        argsorted_affinity=np.argsort(clustering.affinity_matrix_[84])
        
        sorted_distance_labels=[]
        sorted_distances_vals=[]

        for j in tqdm(range(len(Y_new))):
            argsorted_affinity=np.argsort(clustering.affinity_matrix_[84+j])[::-1]
            sorted_affinity=np.sort(clustering.affinity_matrix_[84+j])[::-1]
            labels=[]
            scores=[]
            for i in range(len(argsorted_affinity)):
                if len(labels)==5:
                    break
                if argsorted_affinity[i]<84:
                    labels.append(argsorted_affinity[i])
                    scores.append(sorted_affinity[i])
            sorted_distance_labels.append(labels)
            sorted_distances_vals.append(scores)
                    
                    

        
        
        examples = cls.drop_duplicates(examples, example_template)
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
            ids=sorted_distance_labels[idx]
            shot_idxs_l.append(np.array(ids))
            shot_scores_l.append(np.array(sorted_distances_vals[idx]))
            
                    
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
    
    args = SpectralAffinityScoreSelectorArgs(selector_type=ES.INFLUENCE,n_shots=8)
    #print(args)
    bs_selector = SpectralAffinityScoreSelector.from_examples(args, candidates, example_template, query_examples=test, device=0)
    #print(bs_selector.demo_candidates)
    #print(bs_selector.query2idx)
    print(bs_selector.shot_scores_l)
    print(bs_selector.shot_idxs_l)