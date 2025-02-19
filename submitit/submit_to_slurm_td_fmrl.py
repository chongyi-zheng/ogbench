import os
from pathlib import Path
from datetime import datetime
import platform

import submitit


def main():
    cluster_name = platform.node().split('-')[0]
    if cluster_name == 'adroit':
        log_root_dir = '/home/cz8792/network'
        partition = 'gpu'
        account = None
    elif cluster_name == 'della':
        log_root_dir = '/home/cz8792/gpfs'
        partition = 'gpu-test'
        account = None
    elif cluster_name in ['soak.cs.princeton.edu', 'wash.cs.princeton.edu',
                          'rinse.cs.princeton.edu', 'spin.cs.princeton.edu']:
        log_root_dir = '/n/fs/rl-chongyiz'
        partition = None
        account = 'pnlp'
    else:
        raise NotImplementedError

    executor = submitit.AutoExecutor(folder="/tmp/submitit_logs")  # this path is not actually used.
    executor.update_parameters(
        slurm_name="td_fmrl",
        slurm_time=int(4 * 60),  # minute
        slurm_partition=partition,
        slurm_account=account,
        slurm_nodes=1,
        slurm_ntasks_per_node=1,  # tasks can share nodes
        slurm_cpus_per_task=8,
        slurm_mem="8G",
        slurm_gpus_per_node=1,
        slurm_stderr_to_stdout=True,
        slurm_array_parallelism=24,
    )

    # ddpgbc hyperparameters: discount, alpha, const_std, num_flow_steps, q_agg, critic_layer_norm, distill_type
    with executor.batch():  # job array
        for env_name in [
            # "antmaze-large-navigate-singletask-v0",
            # "humanoidmaze-medium-navigate-singletask-v0",
            # "antsoccer-arena-navigate-singletask-v0"
            # "pen-human-v1",
            # "door-human-v1",
            "cube-single-play-singletask-task2-v0"
        ]:
            for obs_norm_type in ['none']:
                for discount in [0.99]:
                    for alpha in [0.003, 0.03, 0.3]:
                        for const_std in [True, False]:
                            for num_flow_steps in [10]:
                                for critic_loss_type in ['expectile']:
                                    for critic_noise_type in ['normal', 'marginal_state']:
                                        for use_target_critic_vf in [True, False]:
                                            for expectile in [0.9, 0.95, 0.99]:
                                                for q_agg in ['mean']:
                                                    for normalize_q_loss in [True]:
                                                        for reward_type in ['state', 'state_action']:
                                                            for seed in [10]:
                                                                exp_name = f"{datetime.today().strftime('%Y%m%d')}_td_fmrl_{env_name}_obs_norm={obs_norm_type}_alpha={alpha}_num_flow_steps={num_flow_steps}__critic_loss={critic_loss_type}_critic_noise={critic_noise_type}_use_target_critic_vf={use_target_critic_vf}_expectile={expectile}_q_agg={q_agg}_norm_q={normalize_q_loss}_reward={reward_type}"
                                                                log_dir = os.path.expanduser(
                                                                    f"{log_root_dir}/exp_logs/ogbench_logs/td_fmrl/{exp_name}/{seed}")

                                                                # change the log folder of slurm executor
                                                                submitit_log_dir = os.path.join(os.path.dirname(log_dir),
                                                                                                'submitit')
                                                                executor._executor.folder = Path(
                                                                    submitit_log_dir).expanduser().absolute()

                                                                cmds = f"""
                                                                    unset PYTHONPATH;
                                                                    source $HOME/.zshrc;
                                                                    conda activate ogbench;
                                                                    which python;
                                                                    echo $CONDA_PREFIX;
                    
                                                                    echo job_id: $SLURM_ARRAY_JOB_ID;
                                                                    echo task_id: $SLURM_ARRAY_TASK_ID;
                                                                    squeue -j $SLURM_JOB_ID -o "%.18i %.9P %.8j %.8u %.2t %.6D %.5C %.11m %.11l %.12N";
                                                                    echo seed: {seed};
                    
                                                                    export PROJECT_DIR=$PWD;
                                                                    export PYTHONPATH=$HOME/research/ogbench/impls;
                                                                    export PATH="$PATH":"$CONDA_PREFIX"/bin;
                                                                    export CUDA_VISIBLE_DEVICES=0;
                                                                    export MUJOCO_GL=egl;
                                                                    export PYOPENGL_PLATFORM=egl;
                                                                    export EGL_DEVICE_ID=0;
                                                                    export WANDB_API_KEY=bbb3bca410f71c2d7cfe6fe0bbe55a38d1015831;
                                                                    export D4RL_SUPPRESS_IMPORT_ERROR=1;
                                                                    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia;
                            
                                                                    rm -rf {log_dir};
                                                                    mkdir -p {log_dir};
                                                                    python $PROJECT_DIR/impls/main_rl.py \
                                                                        --enable_wandb=1 \
                                                                        --env_name={env_name} \
                                                                        --obs_norm_type={obs_norm_type} \
                                                                        --eval_episodes=50 \
                                                                        --agent=impls/agents/td_fmrl.py \
                                                                        --agent.discount={discount} \
                                                                        --agent.alpha={alpha} \
                                                                        --agent.const_std={const_std} \
                                                                        --agent.num_flow_steps={num_flow_steps} \
                                                                        --agent.critic_loss_type={critic_loss_type} \
                                                                        --agent.critic_noise_type={critic_noise_type} \
                                                                        --agent.use_target_critic_vf={use_target_critic_vf} \
                                                                        --agent.expectile={expectile} \
                                                                        --agent.q_agg={q_agg} \
                                                                        --agent.normalize_q_loss={normalize_q_loss} \
                                                                        --agent.reward_type={reward_type} \
                                                                        --seed={seed} \
                                                                        --save_dir={log_dir} \
                                                                    2>&1 | tee {log_dir}/stream.log;
                    
                                                                    export SUBMITIT_RECORD_FILENAME={log_dir}/submitit_"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID".txt;
                                                                    echo "{submitit_log_dir}/"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"_submitted.pkl" >> "$SUBMITIT_RECORD_FILENAME";
                                                                    echo "{submitit_log_dir}/"$SLURM_ARRAY_JOB_ID"_submission.sh" >> "$SUBMITIT_RECORD_FILENAME";
                                                                    echo "{submitit_log_dir}/"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"_0_log.out" >> "$SUBMITIT_RECORD_FILENAME";
                                                                    echo "{submitit_log_dir}/"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"_0_result.pkl" >> "$SUBMITIT_RECORD_FILENAME";
                                                                """

                                                                cmd_func = submitit.helpers.CommandFunction([
                                                                    "/bin/zsh", "-c",
                                                                    cmds,
                                                                ], verbose=True)

                                                                executor.submit(cmd_func)


if __name__ == "__main__":
    main()
