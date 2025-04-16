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
        nodelist = None
    elif 'della' in cluster_name:
        log_root_dir = '/home/cz8792/gpfs'
        partition = 'gpu-test'
        account = None
        nodelist = None
    elif cluster_name in ['soak.cs.princeton.edu', 'wash.cs.princeton.edu',
                          'rinse.cs.princeton.edu', 'spin.cs.princeton.edu']:
        log_root_dir = '/n/fs/rl-chongyiz'
        partition = None
        account = 'allcs'
        nodelist = "node205,node206,node207"
    elif cluster_name == 'neuronic.cs.princeton.edu':
        log_root_dir = '/n/fs/prl-chongyiz'
        partition = 'all'
        account = None
        nodelist = None
    else:
        raise NotImplementedError

    executor = submitit.AutoExecutor(folder="/tmp/submitit_logs")  # this path is not actually used.
    executor.update_parameters(
        slurm_name="ifac",
        slurm_time=int(12 * 60),  # minute
        slurm_partition=partition,
        slurm_account=account,
        slurm_nodes=1,
        slurm_ntasks_per_node=1,  # tasks can share nodes
        slurm_cpus_per_task=16,
        slurm_mem="64G",
        slurm_gpus_per_node=1,
        slurm_nodelist=nodelist,
        slurm_stderr_to_stdout=True,
        slurm_array_parallelism=20,
    )

    with executor.batch():  # job array
        for env_name in [
            # "antmaze-large-navigate-singletask-v0",
            # "humanoidmaze-medium-navigate-singletask-v0",
            # "antsoccer-arena-navigate-singletask-v0"
            "visual-cube-single-play-singletask-task1-v0",
        ]:
            for obs_norm_type in ['none']:
                for alpha in [3000, 300, 30, 3]:
                    for distill_type in ['fwd_sample']:
                        for value_noise_type in ['normal']:
                            for expectile in [0.85, 0.9, 0.95, 0.99]:
                                for encoder in ['impala_small']:
                                    for q_agg in ['min', 'mean']:
                                        for normalize_q_loss in [False]:
                                            for reward_layer_norm in [False]:
                                                for use_target_reward in [False]:
                                                    for reward_type in ['state']:
                                                        for seed in [20]:
                                                            exp_name = f"{datetime.today().strftime('%Y%m%d')}_ifac_{env_name}_obs_norm={obs_norm_type}_alpha={alpha}_distill={distill_type}_value_noise={value_noise_type}_expectile={expectile}_encoder={encoder}_q_agg={q_agg}_norm_q={normalize_q_loss}_reward_layer_norm={reward_layer_norm}_use_target_reward={use_target_reward}_reward={reward_type}"
                                                            log_dir = os.path.expanduser(
                                                                f"{log_root_dir}/exp_logs/ogbench_logs/ifac/{exp_name}/{seed}")

                                                            # change the log folder of slurm executor
                                                            submitit_log_dir = os.path.join(
                                                                os.path.dirname(log_dir),
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
                                                                    --p_aug=0.5 \
                                                                    --frame_stack=3 \
                                                                    --offline_steps=500_000 \
                                                                    --dataset_class=GCDataset \
                                                                    --agent=impls/agents/ifac.py \
                                                                    --agent.discount=0.99 \
                                                                    --agent.network_type=mlp \
                                                                    --agent.num_residual_blocks=1 \
                                                                    --agent.alpha={alpha} \
                                                                    --agent.num_flow_steps=10 \
                                                                    --agent.distill_type={distill_type} \
                                                                    --agent.value_noise_type={value_noise_type} \
                                                                    --agent.expectile={expectile} \
                                                                    --agent.q_agg={q_agg} \
                                                                    --agent.reward_layer_norm={reward_layer_norm} \
                                                                    --agent.actor_layer_norm=False \
                                                                    --agent.normalize_q_loss={normalize_q_loss} \
                                                                    --agent.use_target_reward={use_target_reward} \
                                                                    --agent.reward_type={reward_type} \
                                                                    --agent.encoder={encoder} \
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
