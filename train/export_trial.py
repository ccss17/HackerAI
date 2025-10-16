import optuna, json, sys

trial_num = 27

study = optuna.load_study(storage='sqlite:///train/optuna_results/study.db', study_name='sft-optuna-search')
trial = [t for t in study.trials if t.number == trial_num][0]

data = {'params': trial.params, 'value': trial.value}
with open(f'train/optuna_results/trial_{trial_num}.json', 'w') as f:
    json.dump(data, f, indent=2)
    
print(f'trial_{trial_num}.json saved')
