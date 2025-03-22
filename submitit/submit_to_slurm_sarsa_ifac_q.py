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
        slurm_name="sarsa_ifac_q",
        slurm_time=int(8 * 60),  # minute
        slurm_partition=partition,
        slurm_account=account,
        slurm_nodes=1,
        slurm_ntasks_per_node=1,  # tasks can share nodes
        slurm_cpus_per_task=8,
        slurm_mem="8G",
        slurm_gpus_per_node=1,
        slurm_stderr_to_stdout=True,
        slurm_array_parallelism=60,
    )

    with executor.batch():  # job array
        for env_name in [
            # "antmaze-large-navigate-singletask-v0",
            # "humanoidmaze-medium-navigate-singletask-v0",
            # "antsoccer-arena-navigate-singletask-v0"
            # "antmaze-medium-play-v2",
            # "pen-human-v1",
            # "door-human-v1",
            "cube-single-play-singletask-task2-v0",
            "cube-double-play-singletask-task2-v0",
            # "cube-triple-play-singletask-task2-v0",
            # "scene-play-singletask-task2-v0",
            # "puzzle-3x3-play-singletask-task4-v0"
            # "cheetah_run",
            # "walker_walk",
        ]:
            for obs_norm_type in ['none', 'normal']:
                for lr in [3e-4]:
                    for batch_size in [256]:
                        for network_size in [512]:
                            for alpha in [300.0, 30.0, 3.0, 0.3]:
                                for num_flow_goals in [16]:
                                    for expectile in [0.75, 0.85, 0.9, 0.95, 0.99]:
                                        for q_agg in ['mean']:
                                            for normalize_q_loss in [False]:
                                                for critic_fm_loss_type in ['sarsa_squared']:
                                                    for reward_type in ['state']:
                                                        for use_reward_func in [True]:
                                                            for use_terminal_masks in [False]:
                                                                for seed in [10]:
                                                                    exp_name = f"{datetime.today().strftime('%Y%m%d')}_sarsa_ifac_q_{env_name}_obs_norm={obs_norm_type}_lr={lr}_bs={batch_size}_ns={network_size}_alpha={alpha}_num_fg={num_flow_goals}_expectile={expectile}_q_agg={q_agg}_norm_q={normalize_q_loss}_critic_fm_loss={critic_fm_loss_type}_reward={reward_type}_use_reward_fn={use_reward_func}_use_mask={use_terminal_masks}"
                                                                    log_dir = os.path.expanduser(
                                                                        f"{log_root_dir}/exp_logs/ogbench_logs/sarsa_ifac_q/{exp_name}/{seed}")

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
                                                                        export WANDB_API_KEY=bbb3bca410f71c2d7cfe6fe0bbe55a38d1015831;
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
                                                                            --agent=impls/agents/sarsa_ifac_q.py \
                                                                            --agent.batch_size={batch_size} \
                                                                            --agent.actor_hidden_dims="({network_size},{network_size},{network_size},{network_size})" \
                                                                            --agent.value_hidden_dims="({network_size},{network_size},{network_size},{network_size})" \
                                                                            --agent.reward_hidden_dims="({network_size},{network_size},{network_size},{network_size})" \
                                                                            --agent.lr={lr} \
                                                                            --agent.tau=0.005 \
                                                                            --agent.network_type=mlp \
                                                                            --agent.num_residual_blocks=1 \
                                                                            --agent.alpha={alpha} \
                                                                            --agent.num_flow_steps=10 \
                                                                            --agent.distill_type=fwd_sample \
                                                                            --agent.critic_noise_type=normal \
                                                                            --agent.critic_fm_loss_type={critic_fm_loss_type} \
                                                                            --agent.num_flow_goals={num_flow_goals} \
                                                                            --agent.clip_flow_goals=False \
                                                                            --agent.ode_solver_type=euler \
                                                                            --agent.expectile={expectile} \
                                                                            --agent.q_agg={q_agg} \
                                                                            --agent.reward_layer_norm=True \
                                                                            --agent.actor_layer_norm=False \
                                                                            --agent.normalize_q_loss={normalize_q_loss} \
                                                                            --agent.use_target_reward=False \
                                                                            --agent.reward_type={reward_type} \
                                                                            --agenr.use_reward_func={use_reward_func} \
                                                                            --agent.use_terminal_masks={use_terminal_masks} \
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
