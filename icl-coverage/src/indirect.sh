#python experiments.py --label 'final' --datasets "indmmlualg;indmmlumedgen;indmmlupropsy;indmmluforlog;indmmlumordis;indmmlupubrel;indmmlucomsec;indmmluast;indmmlunut;indmmluhighbio;indmmlabuseth;indstraqa;indbigshu7;indbigforfal;indbigshu3;indbighyp;indbiglog5" --seeds '0' --selectors "influence;influenceidentity;random;cosine;bertscore" --lms "llama13B" --lm-batch-size 20 --batch-size 20 --n-shots '7' --baselines-exp --paramsfile "params/params-all.jsonl" --run --no-collate-results
#python experiments.py --label 'final' --datasets "indmmlualg;indmmlumedgen;indmmlupropsy;indmmluforlog;indmmlumordis;indmmlupubrel;indmmlucomsec;indmmluast;indmmlunut;indmmluhighbio;indstraqa;indbigshu7;indbigforfal;indbigshu3;indbighyp;indbiglog5" --seeds '2;4' --selectors "random" --lms "llama13B" --lm-batch-size 20 --batch-size 20 --n-shots '5;7' --baselines-exp --paramsfile "params/params-all.jsonl" --run --no-collate-results
#python experiments.py --label 'final' --datasets "indmmlualg;indmmlumedgen;indmmlupropsy;indmmluforlog;indmmlumordis;indmmlupubrel;indmmlucomsec;indmmluast;indmmlunut;indmmluhighbio;indstraqa;indbigshu7;indbigforfal;indbigshu3;indbighyp;indbiglog5" --seeds '0' --selectors "influence;influenceidentity;kmeansclosest;kmeanscentroid;spectralaffinity" --lms "llama13B" --lm-batch-size 20 --batch-size 20 --n-shots '5' --baselines-exp --paramsfile "params/params-all.jsonl" --run --no-collate-results
#python experiments.py --label 'final' --datasets "indmmlupubrel;indmmlucomsec;indmmluast;indmmlunut" --seeds '0' --selectors "influence;influenceidentity" --lms "llama13B" --lm-batch-size 20 --batch-size 20 --n-shots '5' --baselines-exp --paramsfile "params/params-all.jsonl" --run --no-collate-results
#python experiments.py --label 'final' --datasets "indbighyp" --seeds '0' --selectors "random;cosine;bertscore" --lms "llama13B" --lm-batch-size 20 --batch-size 20 --n-shots '7' --baselines-exp --paramsfile "params/params-all.jsonl" --run --no-collate-results
# python experiments.py --label 'final' --datasets "indmmlualg;indmmlumedgen;indmmlupropsy;indmmluforlog;indmmlumordis;indmmlupubrel;indmmlucomsec;indmmluast;indmmlunut;indmmluhighbio;indstraqa;indbigshu7;indbigforfal;indbigshu3;indbighyp;indbiglog5" --seeds '2' --selectors "robertainfluence" --lms "llama13B" --lm-batch-size 20 --batch-size 20 --n-shots '5' --baselines-exp --paramsfile "params/params-all.jsonl" --run --no-collate-results




python experiments.py --label 'final' --datasets "indmmlualg;indmmlumedgen;indmmlupropsy;indmmluforlog;indmmlumordis;indmmlupubrel;indmmlucomsec;indmmluast;indmmlunut;indmmluhighbio;indbigshu7;indbigforfal;indbigshu3;indbiglog5" --seeds '1' --selectors "cosineinfluencepruning;bertscoreinfluencepruning" --lms "llama13B" --lm-batch-size 20 --batch-size 20 --n-shots '5' --baselines-exp --paramsfile "params/params-all.jsonl" --run --no-collate-results --no-coverage-results

#python run.py run-exps-parallel --paramsfile "params/params-all.jsonl" --gpus "0"


#NOISE

# python experiments.py --label 'final' --datasets "mrpc" --seeds '1' --selectors "cosineinfluencepruning" --lms "llama13B" --lm-batch-size 20 --batch-size 20 --n-shots '8' --baselines-exp --paramsfile "params/params-all.jsonl" --run --no-collate-results --no-coverage-results --influence-version "identity"