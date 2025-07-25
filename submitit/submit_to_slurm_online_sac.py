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
        slurm_name="sac",
        slurm_time=int(4 * 60),  # minute
        slurm_partition=partition,
        slurm_account=account,
        slurm_nodes=1,
        slurm_ntasks_per_node=1,  # tasks can share nodes
        slurm_cpus_per_task=8,
        slurm_mem="16G",
        slurm_gpus_per_node=1,
        slurm_stderr_to_stdout=True,
        slurm_array_parallelism=25,
    )

    with executor.batch():  # job array
        for env_name in [
            # ML10 training tasks
            "reach-v2",
            "push-v2",
            "pick-place-v2",
            "door-open-v2",
            "drawer-close-v2",
            # "button-press-topdown-v2",
            # "peg-insert-side-v2",
            # "window-open-v2",
            # "sweep-v2",
            # "basketball-v2",
        ]:
            for train_steps in [1_000_000]:
                for lr in [1e-4, 3e-4]:
                    for layer_norm in [True, False]:
                        for save_final_rb in [1]:
                            for seed in [10, 20]:
                                exp_name = f"{datetime.today().strftime('%Y%m%d')}_sac_{env_name}_train_steps={train_steps}_lr={lr}_layer_norm={layer_norm}_save_final_rb={save_final_rb}_randomize_init_state"
                                log_dir = os.path.expanduser(
                                    f"{log_root_dir}/exp_logs/ogbench_logs/sac/{exp_name}/{seed}")

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
                                    python $PROJECT_DIR/data_gen_scripts/main_sac.py \
                                        --enable_wandb=1 \
                                        --env_name={env_name} \
                                        --seed_steps=10_000 \
                                        --train_steps={train_steps} \
                                        --eval_episodes=50 \
                                        --save_interval=500_000 \
                                        --save_final_replay_buffer={save_final_rb} \
                                        --agent=impls/agents/sac.py \
                                        --agent.lr={lr} \
                                        --agent.actor_hidden_dims="(256,256,256,256)" \
                                        --agent.value_hidden_dims="(256,256,256,256)" \
                                        --agent.layer_norm={layer_norm} \
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
