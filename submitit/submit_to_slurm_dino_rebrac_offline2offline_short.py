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
    elif cluster_name in ['soak.cs.princeton.edu', 'wash.cs.princeton.edu',
                          'rinse.cs.princeton.edu', 'spin.cs.princeton.edu']:
        log_root_dir = '/n/fs/rl-chongyiz'
        partition = None
        account = 'allcs'
        exclude = None
    elif cluster_name == 'neuronic.cs.princeton.edu':
        log_root_dir = '/n/fs/prl-chongyiz'
        partition = 'all'
        account = None
        exclude = 'neu324,neu325,neu329,neu306'
    else:
        raise NotImplementedError

    executor = submitit.AutoExecutor(folder="/tmp/submitit_logs")  # this path is not actually used.
    executor.update_parameters(
        slurm_name="dino_rebrac_offline2offline",
        slurm_time=int(8 * 60),  # minute
        slurm_partition=partition,
        slurm_account=account,
        slurm_nodes=1,
        slurm_ntasks_per_node=1,  # tasks can share nodes
        slurm_cpus_per_task=8,
        slurm_mem="16G",
        slurm_gpus_per_node=1,
        slurm_stderr_to_stdout=True,
        slurm_exclude=exclude,
        slurm_array_parallelism=10,
    )

    # ddpgbc hyperparameters: discount, alpha, num_flow_steps, normalize_q_loss
    with executor.batch():  # job array
        for env_name in [
            # "antmaze-large-navigate-singletask-v0",
            # "humanoidmaze-medium-navigate-singletask-v0",
            # "antsoccer-arena-navigate-singletask-v0"
            # "cube-single-play-singletask-task1-v0",
            # "cube-single-play-singletask-task2-v0",
            # "cube-single-play-singletask-task3-v0",
            # "cube-single-play-singletask-task4-v0",
            # "cube-single-play-singletask-task5-v0",
            # "cube-double-play-singletask-task1-v0",
            # "cube-double-play-singletask-task2-v0",
            # "cube-double-play-singletask-task3-v0",
            # "cube-double-play-singletask-task4-v0",
            # "cube-double-play-singletask-task5-v0",
            # "scene-play-singletask-task1-v0",
            # "scene-play-singletask-task2-v0",
            # "scene-play-singletask-task3-v0",
            # "scene-play-singletask-task4-v0",
            # "scene-play-singletask-task5-v0",
            # "puzzle-4x4-play-singletask-task1-v0",
            # "puzzle-4x4-play-singletask-task2-v0",
            # "puzzle-4x4-play-singletask-task3-v0",
            # "puzzle-4x4-play-singletask-task4-v0",
            # "puzzle-4x4-play-singletask-task5-v0",
            "cheetah_run",
            # "cheetah_run_backward",
            # "cheetah_walk",
            # "cheetah_walk_backward",
            # "walker_walk",
            # "walker_flip",
            # "walker_stand",
            # "walker_run",
            # "quadruped_jump",
            # "quadruped_run",
            # "quadruped_stand",
            # "quadruped_walk",
            # "jaco_reach_top_left",
            # "jaco_reach_top_right",
            # "jaco_reach_bottom_left",
            # "jaco_reach_bottom_right",
        ]:
            for obs_norm_type in ['normal']:
                for finetuning_size in [500_000]:
                    for finetuning_steps in [250_000]:
                        for eval_interval in [1_000]:
                            for alpha_actor in [0.1]:
                                for alpha_critic in [0.1]:
                                    for actor_freq in [4]:
                                        for repr_noise in [0.2]:
                                            for repr_noise_clip in [0.2]:
                                                for repr_temp in [0.1]:
                                                    for target_repr_temp in [0.04]:
                                                        for seed in [100, 200, 300]:
                                                            exp_name = f"{datetime.today().strftime('%Y%m%d')}_dino_rebrac_offline2offline_{env_name}_obs_norm_type={obs_norm_type}_alpha_actor={alpha_actor}_alpha_critic={alpha_critic}_actor_freq={actor_freq}_repr_noise={repr_noise}_repr_noise_clip={repr_noise_clip}_repr_temp={repr_temp}_target_repr_temp={target_repr_temp}"
                                                            log_dir = os.path.expanduser(
                                                                f"{log_root_dir}/exp_logs/ogbench_logs/dino_rebrac_offline2offline/{exp_name}/{seed}")

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
                                                                source $HOME/env_vars.sh
                                                                export D4RL_SUPPRESS_IMPORT_ERROR=1;
                                                                export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia;
                                                                export XLA_FLAGS=--xla_gpu_triton_gemm_any=true;
                            
                                                                rm -rf {log_dir};
                                                                mkdir -p {log_dir};
                                                                python $PROJECT_DIR/impls/main_offline2offline.py \
                                                                    --enable_wandb=1 \
                                                                    --env_name={env_name} \
                                                                    --obs_norm_type={obs_norm_type} \
                                                                    --finetuning_size={finetuning_size} \
                                                                    --finetuning_steps={finetuning_steps} \
                                                                    --eval_interval={eval_interval} \
                                                                    --eval_episodes=50 \
                                                                    --agent=impls/agents/dino_rebrac.py \
                                                                    --agent.discount=0.99 \
                                                                    --agent.alpha_actor={alpha_actor} \
                                                                    --agent.alpha_critic={alpha_critic} \
                                                                    --agent.actor_freq={actor_freq} \
                                                                    --agent.repr_noise={repr_noise} \
                                                                    --agent.repr_noise_clip={repr_noise_clip} \
                                                                    --agent.repr_temp={repr_temp} \
                                                                    --agent.target_repr_temp={target_repr_temp} \
                                                                    --agent.encoder=mlp \
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
