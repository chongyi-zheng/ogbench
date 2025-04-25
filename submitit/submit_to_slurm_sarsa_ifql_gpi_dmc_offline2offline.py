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
        account = 'allcs'
    elif cluster_name == 'neuronic.cs.princeton.edu':
        log_root_dir = '/n/fs/prl-chongyiz'
        partition = 'all'
        account = None
    else:
        raise NotImplementedError

    executor = submitit.AutoExecutor(folder="/tmp/submitit_logs")  # this path is not actually used.
    executor.update_parameters(
        slurm_name="sarsa_ifql_gpi_offline2offline",
        slurm_time=int(12 * 60),  # minute
        slurm_partition=partition,
        slurm_account=account,
        slurm_nodes=1,
        slurm_ntasks_per_node=1,  # tasks can share nodes
        slurm_cpus_per_task=8,
        slurm_mem="16G",
        slurm_gpus_per_node=1,
        slurm_stderr_to_stdout=True,
        slurm_array_parallelism=20,
    )

    with executor.batch():  # job array
        for env_name in [
            # "antmaze-large-navigate-singletask-v0",
            # "humanoidmaze-medium-navigate-singletask-v0",
            # "antsoccer-arena-navigate-singletask-v0"
            # "antmaze-medium-play-v2",
            # "pen-human-v1",
            # "door-human-v1",
            # "cube-single-play-singletask-task2-v0",
            # "cube-double-play-singletask-task2-v0",
            # "scene-play-singletask-task2-v0",
            # "puzzle-3x3-play-singletask-task1-v0"
            "cheetah_run",
            # "cheetah_run_backward",
            # "cheetah_walk",
            # "cheetah_walk_backward",
            "walker_walk",
            # "walker_flip",
            # "walker_stand",
            # "walker_run",
            # "quadruped_run",
            # "quadruped_jump",
            # "quadruped_stand",
            # "quadruped_walk",
            # "jaco_reach_top_left",
            # "jaco_reach_top_right",
            # "jaco_reach_bottom_left",
            # "jaco_reach_bottom_right",
        ]:
            for obs_norm_type in ['normal']:
                for lr in [3e-4]:
                    for tau in [0.005]:  # 1.0 doesn't work better than 0.005
                        for alpha in [0.3]:
                            for num_flow_latents in [1]:
                                for num_flow_goals in [8, 16, 32]:
                                    for actor_freq in [4]:
                                        for expectile in [0.5, 0.8, 0.9, 0.99]:
                                            for vector_field_type in ['mlp']:
                                                for vector_field_time_sin_embedding in [True, False]:
                                                    for clip_flow_goals in [True, False]:
                                                        for use_mixup in [False]:
                                                            for mixup_bw in [0.1]:
                                                                for reward_type in ['state']:
                                                                    for seed in [20]:
                                                                        exp_name = f"{datetime.today().strftime('%Y%m%d')}_sarsa_ifql_gpi_offline2offline_{env_name}_obs_norm={obs_norm_type}_lr={lr}_tau={tau}_alpha={alpha}_num_fl={num_flow_latents}_num_fg={num_flow_goals}_actor_freq={actor_freq}_expectile={expectile}_vf_type={vector_field_type}_vf_time_emb={vector_field_time_sin_embedding}_clip_fgs={clip_flow_goals}_mixup={use_mixup}_mixup_bw={mixup_bw}_reward={reward_type}"
                                                                        log_dir = os.path.expanduser(
                                                                            f"{log_root_dir}/exp_logs/ogbench_logs/sarsa_ifql_gpi_offline2offline/{exp_name}/{seed}")

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
                                                                            python $PROJECT_DIR/impls/main_offline2offline.py \
                                                                                --enable_wandb=1 \
                                                                                --env_name={env_name} \
                                                                                --obs_norm_type={obs_norm_type} \
                                                                                --eval_episodes=50 \
                                                                                --agent=impls/agents/sarsa_ifql_gpi.py \
                                                                                --agent.batch_size=256 \
                                                                                --agent.transition_hidden_dims="(512,512,512,512)" \
                                                                                --agent.actor_hidden_dims="(512,512,512,512)" \
                                                                                --agent.value_hidden_dims="(512,512,512,512)" \
                                                                                --agent.reward_hidden_dims="(512,512,512,512)" \
                                                                                --agent.lr={lr} \
                                                                                --agent.tau={tau} \
                                                                                --agent.network_type=mlp \
                                                                                --agent.num_residual_blocks=1 \
                                                                                --agent.alpha={alpha} \
                                                                                --agent.num_flow_steps=10 \
                                                                                --agent.critic_noise_type=normal \
                                                                                --agent.critic_fm_loss_type=sarsa_squared \
                                                                                --agent.num_flow_latents={num_flow_latents} \
                                                                                --agent.num_flow_goals={num_flow_goals} \
                                                                                --agent.actor_freq={actor_freq} \
                                                                                --agent.clip_flow_goals={clip_flow_goals} \
                                                                                --agent.expectile={expectile} \
                                                                                --agent.vector_field_type={vector_field_type} \
                                                                                --agent.vector_field_time_sin_embedding={vector_field_time_sin_embedding} \
                                                                                --agent.q_agg=min \
                                                                                --agent.reward_layer_norm=True \
                                                                                --agent.actor_layer_norm=False \
                                                                                --agent.normalize_q_loss=False \
                                                                                --agent.use_mixup={use_mixup} \
                                                                                --agent.mixup_alpha=2.0 \
                                                                                --agent.mixup_bandwidth={mixup_bw} \
                                                                                --agent.reward_type={reward_type} \
                                                                                --agent.use_terminal_masks=False \
                                                                                --seed={seed} \
                                                                                --save_dir={log_dir} \
                                                                            2>&1 | tee {log_dir}/stream.log;
    
                                                                            export SUBMITIT_RECORD_FILENAME={log_dir}/submitit_"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID".txt;
                                                                            echo "{submitit_log_dir}/"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"_submitted.pkl" >> "$SUBMITIT_RECORD_FILENAME";
                                                                            echo "{submitit_log_dir}/"$SLURM_ARRAY_JOB_ID"_submission.sh" >> "$SUBMITIT_RECORD_FILENAME";
                                                                            echo "{submitit_log_dir}/"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"_0_log.out" >> "$SUBMITIT_RECORD_FILENAME";
                                                                            echo "{submitit_log_dir}/"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"_0_result.pkl" >> "$SUBMITIT_RECORD_FILENAME";
                                                                        """

                                                                        cmd_func = submitit.helpers.CommandFunction(
                                                                            [
                                                                                "/bin/zsh", "-c",
                                                                                cmds,
                                                                            ], verbose=True)

                                                                        executor.submit(cmd_func)


if __name__ == "__main__":
    main()
