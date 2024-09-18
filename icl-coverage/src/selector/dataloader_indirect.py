import numpy as np
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer
from torch.utils.data import DataLoader



def load_noisy_dataset_by_task(dataset_name='indmmlupubrel', task="mrpc", noise_ratio=0.0):
    
    datasets = load_dataset("glue", task) 
    
    if dataset_name == 'indmmlualg':
        train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
        validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU-algebra-test-100.hf")

    if dataset_name == 'indmmlumedgen':
        train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
        validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_medical_genetics-test-100.hf")

    if dataset_name == 'indmmlupropsy':
        train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
        validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_professional_psychology-test-100.hf")

    if dataset_name == 'indmmluforlog':
        train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
        validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_formal_logic-test-100.hf")

    if dataset_name == 'indmmlumordis':
        train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
        validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_moral_disputes-test-100.hf")

    if dataset_name == 'indmmlupubrel':
        train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
        validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_public_relations-test-100.hf")

    if dataset_name == 'indmmlucomsec':
        train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
        validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_computer_security-test-100.hf")

    if dataset_name == 'indmmluast':
        train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
        validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_astronomy-test-100.hf")

    if dataset_name == 'indmmlunut':
        train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
        validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_nutrition-test-100.hf")

    if dataset_name == 'indmmluhighbio':
        train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
        validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_high_school_biology-test-100.hf")

    if dataset_name == 'indmmlabuseth':
        train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
        validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_business_ethics-test-100.hf")

    if dataset_name == 'indstraqa':
        train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
        validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/StrategyQA-test-100.hf")

    if dataset_name == 'indbigshu7':
        train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
        validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/BIG_tracking_shuffled_objects_seven_objects-test-100.hf")

    if dataset_name == 'indbigforfal':
        train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
        validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/BIG_formal_fallacies-test-100.hf")

    if dataset_name == 'indbigshu3':
        train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
        validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/BIG_tracking_shuffled_objects_three_objects-test-100.hf")

    if dataset_name == 'indbighyp':
        train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
        validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/BIG_hyperbation-test-100.hf")

    if dataset_name == 'indbiglog5':
        train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
        validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/BIG_logical_deduction_five_objects-test-100.hf")
    
    label_to_idx = {
    '(A)': 0,
    '(B)': 1,
    '(C)': 2,
    '(D)': 3,
    '(E)': 4,
    '(F)': 5,
    '(G)': 6,
    }
    
    
    new_labels=[]
    for k,v in enumerate(train_dataset):
        #print(label_to_idx[v['label']])
        new_labels.append(label_to_idx[v['label']])
        
    train_dataset=train_dataset.remove_columns("label").add_column("label", new_labels)
        

    new_labels=[]
    for k,v in enumerate(validation_dataset):
        # print(label_to_idx[v['label']])
        new_labels.append(label_to_idx[v['label']])
    validation_dataset=validation_dataset.remove_columns("label").add_column("label", new_labels)
        
    

    datasets['train'] = train_dataset
    datasets['validation'] = validation_dataset
    datasets['test'] = validation_dataset
    
    noise_index = []
    # print(datasets['train'][0]['label'])
    # print(datasets['validation'][0]['label'])
    # print(datasets['test'][0]['label'])
    
    return datasets, noise_index

def create_dataloaders(model_name_or_path="roberta-large",
                       task="mrpc",
                       noise_ratio=0.0,
                       batch_size=32,
                       dataset_name='indmmlupubrel'):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


    def tokenize_function(examples, max_length=128):
        # max_length=None => use the model max length (it's actually the default)
        
        outputs = tokenizer(examples['text'], truncation=True, max_length=max_length)

        return outputs

    noisy_datasets, noise_index=load_noisy_dataset_by_task(task=task, noise_ratio=noise_ratio, dataset_name=dataset_name)
    tokenized_datasets = noisy_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )


    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")  
        
    train_dataloader = DataLoader(tokenized_datasets["train"],
                                  shuffle=True, 
                                  collate_fn=collate_fn,
                                  batch_size=batch_size)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], 
                                 shuffle=False, 
                                 collate_fn=collate_fn, 
                                 batch_size=batch_size)
    
    return train_dataloader, eval_dataloader, noise_index, tokenized_datasets, collate_fn


