from tqdm import tqdm
import pickle
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model
)
from datasets import Dataset
import evaluate

class LORAEngine(object):
    def __init__(self, 
                model_name_or_path="roberta-large",
                target_modules=["value"],
                train_dataloader=None,
                eval_dataloader=None,
                device="cuda",
                num_epochs=10,
                lr=3e-4,
                low_rank=2,
                task="mrpc"):
        self.model_name_or_path=model_name_or_path
        self.target_modules=target_modules
        self.train_dataloader=train_dataloader
        self.eval_dataloader=eval_dataloader
        self.device=device
        self.num_epochs=num_epochs
        self.lr=lr
        self.task=task
        self.low_rank=low_rank
        
    def build_LORA_model(self):
        '''
        This function fine-tunes a model for classification tasks. 
        For text generation tasks, please see `notebooks/Influential_Data_Identification-Llama2-Math.ipynb`.
        '''
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path,
                                                                        return_dict=True)
        self.model.config.use_cache = False
        self.model.config.pad_token_id = self.model.config.eos_token_id
            
        peft_config = LoraConfig(task_type="SEQ_CLS",
                                 inference_mode=False, 
                                 target_modules=self.target_modules,
                                 r=self.low_rank,
                                 lora_alpha=self.low_rank, 
                                 lora_dropout=0.05)
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def train_LORA_model(self):
        '''
        This function fine-tunes a model for GLUE classification tasks. 
        For text generation tasks, please see `notebooks/Influential_Data_Identification-Llama2-Math.ipynb`.
        '''
        metric = evaluate.load("glue", self.task)
        optimizer = AdamW(params=self.model.parameters(), lr=self.lr)

        # Instantiate scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0.06*(len(self.train_dataloader)*self.num_epochs),
            num_training_steps=(len(self.train_dataloader)*self.num_epochs),
        )

        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            self.model.train()
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                batch.to(self.device)
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            self.model.eval()
            for step, batch in enumerate(tqdm(self.eval_dataloader)):
                batch.to(self.device)
                with torch.no_grad():
                    outputs = self.model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = predictions, batch["labels"]
                metric.add_batch(
                    predictions=predictions,
                    references=references,
                )

            eval_metric = metric.compute()
            print(f"Epoch {(epoch+1)}:", eval_metric)


    def compute_gradient(self, tokenized_datasets, collate_fn):
        train_dataloader_stochastic = DataLoader(tokenized_datasets["train"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        val_dataloader_stochastic = DataLoader(tokenized_datasets["validation"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        # Compute the gradient
        self.model.eval()
        tr_grad_dict = {}
        for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict={}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k]=v.grad.cpu()
                elif 'lora_B' in k:
                    # first index of shape indicates low-rank
                    grad_dict[k]=v.grad.cpu().T
                elif 'modules_to_save.default.out_proj.weight' in k:
                    grad_dict[k]=v.grad.cpu()
                else:
                    pass
            tr_grad_dict[step]=grad_dict
            del grad_dict
            
        val_grad_dict = {}
        for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict={}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k]=v.grad.cpu()
                elif 'lora_B' in k:
                    # first index of shape indicates low-rank
                    grad_dict[k]=v.grad.cpu().T
                elif 'modules_to_save.default.out_proj.weight' in k:
                    grad_dict[k]=v.grad.cpu()
                else:
                    pass
            val_grad_dict[step]=grad_dict    
            del grad_dict
            
        return tr_grad_dict, val_grad_dict


class LORAEngineGeneration(object):
    def __init__(self, 
                base_path,
                project_path,
                dataset_name='math_with_reason',
                device="cuda"):
        self.base_path = base_path
        self.project_path = project_path
        #self.adapter_path = "/nas02/Hadi/Incontenxt-influence/DataInf/Llama13b-hellaswag-justlora" #changed for LESS
        #self.dataset_name = dataset_name
        self.device=device
        self.load_pretrained_network()
        self.load_datasets(dataset_name)   #need to change

    def load_pretrained_network(self):
        # setup tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(self.base_path)
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # load a base model
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, load_in_4bit=False)
        base_model = LlamaForCausalLM.from_pretrained(
            self.base_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            offload_folder="offload",
            offload_state_dict=True,
            device_map='auto'
        )

        # load a pre-trained model.
        # self.model = PeftModel.from_pretrained(base_model, self.adapter_path, is_trainable=True)
        # self.finetuned_config = LoraConfig.from_pretrained(pretrained_model_name_or_path=self.adapter_path)  #Changed for LESS
        self.model=base_model

    def load_datasets(self,dataset_name):
        # print('qnli')
        if dataset_name == 'indmmlualg':
            self.train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
            self.validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU-algebra-test-100.hf")

        if dataset_name == 'indmmlumedgen':
            self.train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
            self.validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_medical_genetics-test-100.hf")

        if dataset_name == 'indmmlupropsy':
            self.train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
            self.validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_professional_psychology-test-100.hf")

        if dataset_name == 'indmmluforlog':
            self.train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
            self.validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_formal_logic-test-100.hf")

        if dataset_name == 'indmmlumordis':
            self.train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
            self.validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_moral_disputes-test-100.hf")

        if dataset_name == 'indmmlupubrel':
            self.train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
            self.validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_public_relations-test-100.hf")

        if dataset_name == 'indmmlucomsec':
            self.train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
            self.validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_computer_security-test-100.hf")

        if dataset_name == 'indmmluast':
            self.train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
            self.validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_astronomy-test-100.hf")

        if dataset_name == 'indmmlunut':
            self.train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
            self.validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_nutrition-test-100.hf")

        if dataset_name == 'indmmluhighbio':
            self.train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
            self.validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_high_school_biology-test-100.hf")

        if dataset_name == 'indmmlabuseth':
            self.train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
            self.validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/MMLU_business_ethics-test-100.hf")

        if dataset_name == 'indstraqa':
            self.train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
            self.validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/StrategyQA-test-100.hf")

        if dataset_name == 'indbigshu7':
            self.train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
            self.validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/BIG_tracking_shuffled_objects_seven_objects-test-100.hf")

        if dataset_name == 'indbigforfal':
            self.train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
            self.validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/BIG_formal_fallacies-test-100.hf")

        if dataset_name == 'indbigshu3':
            self.train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
            self.validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/BIG_tracking_shuffled_objects_three_objects-test-100.hf")

        if dataset_name == 'indbighyp':
            self.train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
            self.validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/BIG_hyperbation-test-100.hf")

        if dataset_name == 'indbiglog5':
            self.train_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/COT-train-84-new.hf")
            self.validation_dataset = Dataset.load_from_disk("/nas02/Hadi/Incontenxt-influence/DataInf/Indirect-ICL/datasets/BIG_logical_deduction_five_objects-test-100.hf")


    def create_tokenized_datasets(self):
        tokenize_func = lambda x: self.tokenizer(
            x["text"], truncation=True, padding=True, max_length=128, return_tensors="pt" # text should be more appropritate
        ).to(self.device)

        # if 'with_reason' in self.dataset_name:
        #     column_list=["text", "answer", "variation", "prompt", "reason"]
        # else:
        #     column_list=["text", "answer", "variation", "prompt"]
        
        # column_list_train=['question', 'sentence', 'label', 'idx', 'text', 'prompt'] # QNLI
        # column_list_test=['question', 'sentence', 'label', 'idx', 'text', 'prompt']
        
        # column_list_train=['article', 'label', 'text',"prompt"]  #AGNEWS
        # column_list_test=['article', 'label', 'text',"prompt"]
        
        # column_list_train=['question', 'answer', 'text']
        # column_list_test=['question', 'answer', 'text']
        
        # column_list_train=['premise', 'hypothesis', 'label', 'idx', 'text', 'prompt']  #MNLI
        # column_list_test=['premise', 'hypothesis', 'label', 'idx', 'text', 'prompt']
        
                
        # column_list_train=['question', 'answer', 'text', 'prompt']  #GSM8k
        # column_list_test=['question', 'answer', 'text', 'prompt']
        
        # column_list_train=['sentence', 'label', 'idx', 'text', 'prompt']  #SST2
        # column_list_test=['sentence', 'label', 'idx', 'text', 'prompt'] 
        
        
        # column_list_train=['question1', 'question2', 'label', 'idx', 'text', 'prompt']  #QQP
        # column_list_test=['question1', 'question2', 'label', 'idx', 'text', 'prompt']
        
        # column_list_train=['ind', 'activity_label', 'ctx_a', 'ctx_b', 'ctx', 'endings', 'source_id', 'split', 'split_type', 'label', 'context', 'text', 'prompt'] #hellaswag
        # column_list_test=['ind', 'activity_label', 'ctx_a', 'ctx_b', 'ctx', 'endings', 'source_id', 'split', 'split_type', 'label', 'context', 'text', 'prompt']

        # column_list_train=['id', 'question', 'question_concept', 'choices', 'answerKey', 'context', 'text', 'prompt']  #cmsqa
        # column_list_test=['id', 'question', 'question_concept', 'choices', 'answerKey', 'context', 'text', 'prompt']
        
        # column_list_train=['question', 'subject', 'choices', 'answer', 'context', 'text', 'prompt']  #mmlu
        # column_list_test=['question', 'subject', 'choices', 'answer', 'context', 'text', 'prompt']
        
        # column_list_train=['text_old', 'label', 'label_text', 'context', 'text', 'prompt']  #banking77
        # column_list_test=['text_old', 'label', 'label_text', 'context', 'text', 'prompt']
        
        # column_list_train=['original_text', 'label', 'context', 'text', 'prompt', 'label_text'] #sarcasm
        # column_list_test=['original_text', 'label', 'context', 'text', 'prompt', 'label_text']
        
        # column_list_train=['original_text', 'label', 'context', 'text', 'prompt', 'label_text'] #tweetevalirony
        # column_list_test=['original_text', 'label', 'context', 'text', 'prompt', 'label_text']
        
        column_list_train=['text', 'label'] #indmmlualgebra
        column_list_test=['text', 'label']
        
        tokenized_datasets=dict()
        tokenized_datasets["train"] = self.train_dataset.map(
            tokenize_func,
            batched=True,
            remove_columns=column_list_train,
        )
        tokenized_datasets["validation"] = self.validation_dataset.map(
            tokenize_func,
            batched=True,
            remove_columns=column_list_test,
        )
        collate_fn = lambda x: self.tokenizer.pad(x, padding="longest", return_tensors="pt")
        
        # tokenized_datasets['train'] = tokenized_datasets['train'].select([*range(0,900,1)])  #pakro
        # tokenized_datasets['validation'] = tokenized_datasets['validation'].select([*range(0,100,1)])

        
        return tokenized_datasets, collate_fn

    # def compute_gradient(self, tokenized_datasets, collate_fn):
    #     train_dataloader_stochastic = DataLoader(tokenized_datasets["train"], 
    #                                               shuffle=False,
    #                                               collate_fn=collate_fn,
    #                                               batch_size=1)
    #     val_dataloader_stochastic = DataLoader(tokenized_datasets["validation"], 
    #                                               shuffle=False,
    #                                               collate_fn=collate_fn,
    #                                               batch_size=1)
        
    
    #     # Compute the gradient
    #     self.model.eval()
    #     tr_grad_dict = {}
    #     for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
    #         self.model.zero_grad() # zeroing out gradient
    #         batch['labels'] = batch['input_ids']
    #         batch.to(self.device)
    #         outputs = self.model(**batch)
    #         loss = outputs.loss
    #         loss.backward()
            
    #         grad_dict={}
    #         for k, v in self.model.named_parameters():
    #             #print(k)
    #             if 'lora_A' in k:
    #             # if 'embed_tokens' in k:
    #                 # print(v)
    #                 # print(v.grad)
    #                 grad_dict[k]=v.grad.cpu()
    #                 # print(k)
    #                 if '.0.' in k:
    #                     # print(k)
    #                     grad_dict[k]=v.grad.cpu()
    #                 elif '.1.' in k:
    #                     # print(k)
    #                     grad_dict[k]=v.grad.cpu()
    #                 elif '38' in k:
    #                     # print(k)
    #                     grad_dict[k]=v.grad.cpu()
    #                 elif '39' in k:
    #                     # print(k)
    #                     grad_dict[k]=v.grad.cpu()
    #                 else:
    #                     pass
                        
    #             elif 'lora_B' in k:
    #                 # print(k)
    #                 # first index of shape indicates low-rank
    #                 if '.0.' in k:
    #                     # print(k)
    #                     grad_dict[k]=v.grad.cpu().T
    #                 elif '.1.' in k:
    #                     # print(k)
    #                     grad_dict[k]=v.grad.cpu().T
    #                 elif '38' in k:
    #                     # print(k)
    #                     grad_dict[k]=v.grad.cpu().T
    #                 elif '39' in k:
    #                     # print(k)
    #                     grad_dict[k]=v.grad.cpu().T
    #                 # elif '.2.' in k:
    #                 #     print(k)
    #                     # grad_dict[k]=v.grad.cpu().T
    #                 else:
    #                     pass
    #             else:
    #                 pass
    #         tr_grad_dict[step]=grad_dict
    #         del grad_dict
        
        
        
    #     val_grad_dict = {}
    #     for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
    #         self.model.zero_grad() # zeroing out gradient
    #         batch['labels'] = batch['input_ids']
    #         batch.to(self.device)
    #         outputs = self.model(**batch)
    #         loss = outputs.loss
    #         loss.backward()
            
    #         grad_dict={}
    #         for k, v in self.model.named_parameters():
    #             #print(k)
    #             if 'lora_A' in k:
    #             # if 'embed_tokens' in k:
    #                 # print(v)
    #                 # print(v.grad)
    #                 grad_dict[k]=v.grad.cpu()
    #                 # print(k)
    #                 if '.0.' in k:
    #                     # print(k)
    #                     grad_dict[k]=v.grad.cpu()
    #                 elif '.1.' in k:
    #                     # print(k)
    #                     grad_dict[k]=v.grad.cpu()
    #                 elif '38' in k:
    #                     # print(k)
    #                     grad_dict[k]=v.grad.cpu()
    #                 elif '39' in k:
    #                     # print(k)
    #                     grad_dict[k]=v.grad.cpu()
    #                 else:
    #                     pass
                        
    #             elif 'lora_B' in k:
    #                 # print(k)
    #                 # first index of shape indicates low-rank
    #                 if '.0.' in k:
    #                     # print(k)
    #                     grad_dict[k]=v.grad.cpu().T
    #                 elif '.1.' in k:
    #                     # print(k)
    #                     grad_dict[k]=v.grad.cpu().T
    #                 elif '38' in k:
    #                     # print(k)
    #                     grad_dict[k]=v.grad.cpu().T
    #                 elif '39' in k:
    #                     # print(k)
    #                     grad_dict[k]=v.grad.cpu().T
    #                 # elif '.2.' in k:
    #                 #     print(k)
    #                     # grad_dict[k]=v.grad.cpu().T
    #                 else:
    #                     pass
    #             else:
    #                 pass
    #         val_grad_dict[step]=grad_dict    
    #         del grad_dict
            
    #     return tr_grad_dict, val_grad_dict
    
    def compute_gradient(self, tokenized_datasets, collate_fn):    #LESS
        train_dataloader_stochastic = DataLoader(tokenized_datasets["train"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        val_dataloader_stochastic = DataLoader(tokenized_datasets["validation"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        # Compute the gradient
        self.model.eval()
        tr_grad_dict = {}
        for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
            # if step==10:
            #     break
            self.model.zero_grad() # zeroing out gradient

            batch['labels'] = batch['input_ids']
            batch.to(self.device)
            loss = self.model(**batch).loss
            loss.backward()
            
            grad_dict={}
            # vectorized_grads = torch.cat(
            #     [p.grad.view(-1) for k,p in self.model.named_parameters() if p.grad is not None])
            for k, v in self.model.named_parameters():
                # print(k)
                # print("Shape of layer {} is {}".format(k,v.shape))
                if v.grad is not None and 'layernorm' in k and (".0." in k or ".1." in k or "38" in k or "39" in k):  # ("37" in k or "38" in k or "39" in k)
                    # print(v.grad.view(-1).cpu().shape)
                    # print(v.grad.view(-1).cpu())
                    # print(v)
                    # print(v.grad)
                    grad_dict[k]=v.grad.view(-1).cpu()
  
                    
                    
            tr_grad_dict[step]=grad_dict
            
            del grad_dict
            
            
        val_grad_dict = {}
        for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
            # if step==10:
            #     break
            self.model.zero_grad() # zeroing out gradient
 
            batch['labels'] = batch['input_ids']
            batch.to(self.device)
            loss = self.model(**batch).loss
            loss.backward()
            
            grad_dict={}
            # vectorized_grads = torch.cat(
            #     [p.grad.view(-1) for k,p in self.model.named_parameters() if p.grad is not None])
            for k, v in self.model.named_parameters():
                if v.grad is not None and 'layernorm' in k and (".0." in k or ".1." in k or "38" in k or "39" in k):  # ("37" in k or "38" in k or "39" in k)
                    grad_dict[k]=v.grad.view(-1).cpu()
                    
            val_grad_dict[step]=grad_dict
            
        return tr_grad_dict, val_grad_dict

