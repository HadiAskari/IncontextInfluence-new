from transformers import AutoTokenizer, LlamaForCausalLM,BitsAndBytesConfig,AutoConfig,AutoModelForCausalLM
import datasets
import transformers
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from tqdm.auto import tqdm
import pickle as pkl

# test=datasets.load_from_disk('datasets/qnli-similar-test-100.hf')
# train=datasets.load_from_disk('datasets/qnli-similar-train-900.hf')

test=datasets.load_from_disk('datasets/isarcasm-test-100.hf')
#original

model = LlamaForCausalLM.from_pretrained("llama-2-13b-chat-converted"
                                          , device_map='auto',
                                          torch_dtype=torch.float16,
                                          #load_in_8bit=True
                                        #   quantization_config=bnb_config
                                        )
tokenizer = AutoTokenizer.from_pretrained("llama-2-13b-chat-converted",use_fast=True)

def get_accuracy(model, tokenizer,test):
    responses=[]
    for sentence,label in tqdm(zip(test['original_text'],test['label_text'])):
        prompt = '<s>[INST]### Human: Answer with Yes or No. Is this sentence sarcastic or not? Sentence: {}. [/INST]\n### Assistant: </s>'.format(sentence)
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

        # Generate
        generate_ids = model.generate(inputs.input_ids, max_new_tokens = 25, temperature=0.01) #.to('cuda')
        responses.append(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

    print(responses)
    with open('sarcasm_original_responses.pkl', 'wb') as f:
        pkl.dump(responses,f)

# def get_accuracy(model, tokenizer,test):
#     responses=[]
#     for sentence,question in tqdm(zip(test['sentence'],test['question'])):
#         prompt = "### Human: Answer the following questions given the passage as Yes or No:{} Can we know {}?### Assistant:".format(sentence,question) 
#         inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

#         # Generate
#         generate_ids = model.generate(inputs.input_ids, max_new_tokens = 4, temperature=0.01) #.to('cuda')
#         responses.append(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

#     print(responses)
#     with open('qnli_original_responses.pkl', 'wb') as f:
#         pkl.dump(responses,f)
        
    # accuracy=0
    # for orig, pred in zip(test['label'], responses):
    #     pred=pred.split('\n')[-1].strip()
    #     # print(pred)
    #     if orig==1:
    #         orig='Yes'
    #     else:
    #         orig='No'
    #     if pred==orig:
    #         accuracy+=1

    # print(responses[0:10])
    
    # print(accuracy)
    # return accuracy


orig_accuracy=get_accuracy(model, tokenizer,test)

# model = LlamaForCausalLM.from_pretrained("Llama2-70b-mtop-combined"
#                                           , device_map='auto',
#                                           torch_dtype=torch.float16,
#                                           #load_in_8bit=True
#                                         #   quantization_config=bnb_config
#                                         )
# tokenizer = AutoTokenizer.from_pretrained("Llama2-70b-mtop-combined",use_fast=True)

# finetuned_accuracy=get_accuracy(model, tokenizer,test)

# accuracies=[orig_accuracy,finetuned_accuracy]

# with open('MTOP-accuracies', 'wb') as f:
#     pkl.dump(accuracies,f)
