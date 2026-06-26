#!/bin/sh
env="bot"
scenario="specific_goal_two_bots"
num_agents=4
algo="rmappo"
exp="check"
save_gifs= False
use_render = False
run=run360
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python test_bot.py --save_gifs  ${save_gifs} --share_policy --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
--use_render ${use_render} --episode_length 240 --render_episodes 500 \
    --model_dir "/home/caid/takieddine/iGibson/igibson/onpolicy/scripts/results/${env}/${scenario}/rmappo/check/${run}/models"
done
