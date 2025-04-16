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
        slurm_name="mcfac",
        slurm_time=int(4 * 60),  # minute
        slurm_partition=partition,
        slurm_account=account,
        slurm_nodes=1,
        slurm_ntasks_per_node=1,  # tasks can share nodes
        slurm_cpus_per_task=16,
        slurm_mem="64G",
        slurm_gpus_per_node=1,
        slurm_stderr_to_stdout=True,
        slurm_array_parallelism=24,
    )

    with executor.batch():  # job array
        for env_name in [
            # "antmaze-large-navigate-singletask-v0",
            # "humanoidmaze-medium-navigate-singletask-v0",
            # "antsoccer-arena-navigate-singletask-v0"
            # "pen-human-v1",
            # "door-human-v1",
            # "cube-single-play-singletask-task2-v0",
            # "cube-double-play-singletask-task2-v0",
            # "scene-play-singletask-task2-v0",
            "visual-cube-single-play-singletask-task1-v0",
        ]:
            for obs_norm_type in ['none']:
                for discount in [0.99]:
                    for alpha in [30, 3]:
                        for distill_type in ['fwd_int']:
                            for critic_loss_type in ['expectile']:
                                for critic_noise_type in ['normal']:
                                    for expectile in [0.9, 0.95, 0.99]:
                                        for q_agg in ['mean', 'min']:
                                            for normalize_q_loss in [True]:
                                                for reward_type in ['state']:
                                                    for encoder in ['impala_small']:
                                                        for encoder_actor_loss_grad in [True, False]:
                                                            for seed in [10]:
                                                                exp_name = f"{datetime.today().strftime('%Y%m%d')}_mcfac_{env_name}_obs_norm={obs_norm_type}_alpha={alpha}_distill={distill_type}_critic_loss={critic_loss_type}_critic_noise={critic_noise_type}_expectile={expectile}_q_agg={q_agg}_norm_q={normalize_q_loss}_reward={reward_type}_encoder={encoder}_encoder_actor_loss_grad={encoder_actor_loss_grad}"
                                                                log_dir = os.path.expanduser(
                                                                    f"{log_root_dir}/exp_logs/ogbench_logs/mcfac/{exp_name}/{seed}")

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
                                                                    python $PROJECT_DIR/impls/main_rl.py \
                                                                        --enable_wandb=1 \
                                                                        --env_name={env_name} \
                                                                        --obs_norm_type={obs_norm_type} \
                                                                        --eval_episodes=50 \
                                                                        --dataset_class=GCDataset \
                                                                        --p_aug=0.5 \
                                                                        --frame_stack=3 \
                                                                        --offline_steps=500_000 \
                                                                        --agent=impls/agents/mcfac.py \
                                                                        --agent.discount={discount} \
                                                                        --agent.alpha={alpha} \
                                                                        --agent.num_flow_steps=10 \
                                                                        --agent.distill_type={distill_type} \
                                                                        --agent.critic_loss_type={critic_loss_type} \
                                                                        --agent.critic_noise_type={critic_noise_type} \
                                                                        --agent.expectile={expectile} \
                                                                        --agent.q_agg={q_agg} \
                                                                        --agent.actor_layer_norm=False \
                                                                        --agent.normalize_q_loss={normalize_q_loss} \
                                                                        --agent.reward_type={reward_type} \
                                                                        --agent.encoder={encoder} \
                                                                        --agent.encoder_actor_loss_grad={encoder_actor_loss_grad} \
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
