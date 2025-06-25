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
    elif cluster_name == 'neuronic.cs.princeton.edu':
        log_root_dir = '/n/fs/prl-chongyiz'
        partition = 'all'
        account = None
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
        slurm_cpus_per_task=16,
        slurm_mem="16G",
        slurm_gpus_per_node=1,
        slurm_stderr_to_stdout=True,
        slurm_array_parallelism=40,
    )

    with executor.batch():  # job array
        for env_name in ["humanoidmaze-medium-navigate-singletask-task3-v0"]:
            for discount in [0.995]:
                for alpha_critic in [0.3, 1, 3]:
                    for alpha_actor in [10, 30, 50]:
                        for value_layer_norm in [False]:
                            for actor_layer_norm in [True]:
                                for seed in [10, 20]:
                                    exp_name = f"{datetime.today().strftime('%Y%m%d')}_fdrl_{env_name}_discount={discount}_alpha_critic={alpha_critic}_alpha_actor={alpha_actor}_value_layer_norm={value_layer_norm}_actor_layer_norm={actor_layer_norm}_fitting_target_q_and_min_bootstrapped_vf_fql_actor_loss_vf_at_t=0_bounded_returns_indep_q_noises"
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
                                            --agent.num_samples=32 \
                                            --agent.num_flow_steps=10 \
                                            --agent.tau=0.005 \
                                            --agent.ode_solver=euler \
                                            --agent.alpha_critic={alpha_critic} \
                                            --agent.alpha_actor={alpha_actor} \
                                            --agent.value_dropout_rate=0.0 \
                                            --agent.value_layer_norm={value_layer_norm} \
                                            --agent.actor_layer_norm={actor_layer_norm} \
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
