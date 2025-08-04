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
        exclude = None
    elif cluster_name == 'della':
        log_root_dir = '/home/cz8792/gpfs'
        partition = 'gpu-test'
        account = None
        exclude = None
    elif cluster_name in ['soak.cs.princeton.edu', 'wash.cs.princeton.edu',
                          'rinse.cs.princeton.edu', 'spin.cs.princeton.edu',
                          'node030.ionic.cs.princeton.edu', 'node202.ionic.cs.princeton.edu']:
        log_root_dir = '/n/fs/rl-chongyiz'
        partition = None
        account = 'pnlp'
        exclude = None
    elif cluster_name == 'neuronic.cs.princeton.edu':
        log_root_dir = '/n/fs/prl-chongyiz'
        partition = 'all'
        account = None
        exclude = 'neu306'
    else:
        raise NotImplementedError

    executor = submitit.AutoExecutor(folder="/tmp/submitit_logs")  # this path is not actually used.
    executor.update_parameters(
        slurm_name="fdrl",
        slurm_time=int(8 * 60),  # minute
        slurm_partition=partition,
        slurm_account=account,
        slurm_nodes=1,
        slurm_ntasks_per_node=1,  # tasks can share nodes
        slurm_cpus_per_task=8,
        slurm_mem="16G",
        slurm_gpus_per_node=1,
        slurm_exclude=exclude,
        slurm_stderr_to_stdout=True,
        slurm_array_parallelism=20,
    )

    with executor.batch():  # job array
        for env_name in ["puzzle-4x4-play-singletask-task4-v0"]:
            for discount in [0.99]:
                for alpha_critic in [0.01, 0.03, 0.1]:
                    for alpha_actor in [300]:
                        for critic_loss_type in ['q-learning']:
                            for next_action_extraction in ['fql']:
                                for policy_extraction in ['fql']:
                                    for ret_agg in ['min', 'mean']:
                                        for q_agg in ['mean']:
                                            for ensemble_weight_temp in [0.1, 10.0, 20.0, 50.0, 100.0]:
                                                for value_layer_norm in [True]:
                                                    for actor_layer_norm in [True]:
                                                        for seed in [10, 20, 30]:
                                                            exp_name = f"{datetime.today().strftime('%Y%m%d')}_fdrl_{env_name}_discount={discount}_alpha_critic={alpha_critic}_alpha_actor={alpha_actor}_critic_loss_type={critic_loss_type}_next_a_extrac={next_action_extraction}_pi_extrac={policy_extraction}_ensem_w_temp={ensemble_weight_temp}_value_ln={value_layer_norm}_actor_ln={actor_layer_norm}_ret_agg={ret_agg}_q_agg={q_agg}"
                                                            log_dir = os.path.expanduser(
                                                                f"{log_root_dir}/exp_logs/fdrl_logs/fdrl/{exp_name}/{seed}")
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
                                                                export PYTHONPATH=$HOME/research/ogbench;
                                                                export PATH="$PATH":"$CONDA_PREFIX"/bin;
                                                                export CUDA_VISIBLE_DEVICES=0;
                                                                export MUJOCO_GL=egl;
                                                                export PYOPENGL_PLATFORM=egl;
                                                                export EGL_DEVICE_ID=0;
                                                                source $HOME/env_vars.sh;
                                                                export D4RL_SUPPRESS_IMPORT_ERROR=1;
                                                                export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia;
                                                                export XLA_FLAGS=--xla_gpu_triton_gemm_any=true;
    
                                                                rm -rf {log_dir};
                                                                mkdir -p {log_dir};
                                                                python $PROJECT_DIR/main.py \
                                                                    --enable_wandb=1 \
                                                                    --env_name={env_name} \
                                                                    --train_steps=1_000_000 \
                                                                    --log_interval=5_000 \
                                                                    --eval_interval=100_000 \
                                                                    --save_interval=1_000_000 \
                                                                    --eval_episodes=50 \
                                                                    --agent=agents/fdrl.py \
                                                                    --agent.discount={discount} \
                                                                    --agent.num_samples=32 \
                                                                    --agent.num_flow_steps=10 \
                                                                    --agent.tau=0.005 \
                                                                    --agent.ode_solver=euler \
                                                                    --agent.alpha_critic={alpha_critic} \
                                                                    --agent.alpha_actor={alpha_actor} \
                                                                    --agent.critic_loss_type={critic_loss_type} \
                                                                    --agent.next_action_extraction={next_action_extraction} \
                                                                    --agent.policy_extraction={policy_extraction} \
                                                                    --agent.ensemble_weight_temp={ensemble_weight_temp} \
                                                                    --agent.value_dropout_rate=0.0 \
                                                                    --agent.value_layer_norm={value_layer_norm} \
                                                                    --agent.actor_layer_norm={actor_layer_norm} \
                                                                    --agent.ret_agg={ret_agg} \
                                                                    --agent.q_agg={q_agg} \
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
