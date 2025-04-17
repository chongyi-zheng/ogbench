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
        # account = 'allcs'
        # nodelist = "node205,node206,node207"
        account = 'pnlp'
        nodelist = "node210,node211"
    elif cluster_name == 'neuronic.cs.princeton.edu':
        log_root_dir = '/n/fs/prl-chongyiz'
        partition = 'all'
        account = None
        nodelist = None
    else:
        raise NotImplementedError

    executor = submitit.AutoExecutor(folder="/tmp/submitit_logs")  # this path is not actually used.
    executor.update_parameters(
        slurm_name="iql_offline2offline",
        slurm_time=int(20 * 60),  # minute
        slurm_partition=partition,
        slurm_account=account,
        slurm_nodes=1,
        slurm_ntasks_per_node=1,  # tasks can share nodes
        slurm_cpus_per_task=8,
        slurm_mem="160G",
        slurm_gpus_per_node=4,
        slurm_nodelist=nodelist,
        slurm_stderr_to_stdout=True,
        slurm_array_parallelism=20,
    )

    with executor.batch():  # job array
        for env_name in [
            "google_robot_pick_coke_can",
        ]:
            for batch_size in [32]:
                for alpha in [1.0, 0.1]:
                    for expectile in [0.9]:
                        for actor_freq in [4]:
                            for seed in [10, 20]:
                                exp_name = f"{datetime.today().strftime('%Y%m%d')}_iql_{env_name}_batch_size={batch_size}_alpha={alpha}_expectile={expectile}_actor_freq={actor_freq}"
                                log_dir = os.path.expanduser(
                                    f"{log_root_dir}/exp_logs/ogbench_logs/iql_octo/{exp_name}/{seed}")

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
                                    export PYTHONPATH=$HOME/research/ogbench/impls:$HOME/research/SimplerEnv;
                                    export PATH="$PATH":"$CONDA_PREFIX"/bin;
                                    export CUDA_VISIBLE_DEVICES=0,1;
                                    export MUJOCO_GL=egl;
                                    export PYOPENGL_PLATFORM=egl;
                                    export EGL_DEVICE_ID=0;
                                    source $HOME/env_vars.sh
                                    export HF_HOME={log_root_dir}/huggingface_cache;
                                    export D4RL_SUPPRESS_IMPORT_ERROR=1;
                                    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia;
                                    export XLA_FLAGS=--xla_gpu_triton_gemm_any=true;
                                    export XLA_PYTHON_CLIENT_PREALLOCATE=false;

                                    rm -rf {log_dir};
                                    mkdir -p {log_dir};
                                    python $PROJECT_DIR/impls/main_octo_offline2offline.py \
                                        --enable_wandb=1 \
                                        --env_name={env_name} \
                                        --eval_episodes=50 \
                                        --pretraining_steps=300_000 \
                                        --finetuning_steps=100_000 \
                                        --eval_interval=20_000 \
                                        --octo=octo_utils/octo_pretrain_config.py:vit_s \
                                        --octo.dataset_kwargs.oxe_kwargs.data_dir={log_root_dir}/datasets/octo_datasets \
                                        --octo.dataset_kwargs.oxe_kwargs.data_mix=google_robot \
                                        --octo.dataset_kwargs.batch_size={batch_size} \
                                        --agent=impls/octo_agents/iql.py \
                                        --agent.discount=0.99 \
                                        --agent.expectile={expectile} \
                                        --agent.alpha={alpha} \
                                        --agent.actor_freq={actor_freq} \
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
