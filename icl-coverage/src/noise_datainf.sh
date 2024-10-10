#for 10%Prune and 0.5/0.5 reweighting on mrpc 0.2


#python experiments.py --label 'final' --datasets "mrpc" --seeds '1' --selectors "random;bertscore;cosine;bertscoreinfluencepruning;cosineinfluencepruning;bertscorerandompruning;cosinerandompruning;bertscoreinfluencereweighting;cosineinfluencereweighting" --lms "llama13B" --lm-batch-size 20 --batch-size 20 --n-shots '8' --baselines-exp --paramsfile "params/params-all.jsonl" --run --no-collate-results --no-coverage-results --influence-version "identity"
#python experiments.py --label 'final' --datasets "mrpc" --seeds '1' --selectors "bertscoreinfluencepruning;cosineinfluencepruning;bertscoreinfluencereweighting;cosineinfluencereweighting" --lms "llama13B" --lm-batch-size 20 --batch-size 20 --n-shots '8' --baselines-exp --paramsfile "params/params-all.jsonl" --run --no-collate-results --no-coverage-results --influence-version "proposed"
# python experiments.py --label 'final' --datasets "mrpc" --seeds '1' --selectors "bertscoreinfluencepruning;cosineinfluencepruning;bertscoreinfluencereweighting;cosineinfluencereweighting" --lms "llama13B" --lm-batch-size 20 --batch-size 20 --n-shots '8' --baselines-exp --paramsfile "params/params-all.jsonl" --run --no-collate-results --no-coverage-results --influence-version "LiSSA"

#python experiments.py --label 'final' --datasets "mrpc" --seeds '1' --selectors "bertscorerandompruning;cosinerandompruning" --lms "llama13B" --lm-batch-size 20 --batch-size 20 --n-shots '8' --baselines-exp --paramsfile "params/params-all.jsonl" --run --no-collate-results --no-coverage-results --influence-version "identity"


#python experiments.py --label 'final' --datasets "mrpc" --seeds '0' --selectors "cosineinfluencereweighting" --lms "llama13B" --lm-batch-size 20 --batch-size 20 --n-shots '8' --baselines-exp --paramsfile "params/params-all.jsonl" --run --no-collate-results --no-coverage-results --influence-version "LiSSA"
python experiments.py --label 'final' --datasets "qnli" --seeds '0' --selectors "bertscoreinfluencepruning;cosineinfluencepruning;bertscoreinfluencereweighting;cosineinfluencereweighting" --lms "llama13B" --lm-batch-size 20 --batch-size 20 --n-shots '8' --baselines-exp --paramsfile "params/params-all.jsonl" --run --no-collate-results --no-coverage-results --influence-version "proposed"

#CUDA_VISIBLE_DEVICES=4,5,6,7 
#random;bertscore;cosine;