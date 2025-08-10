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
        account = 'allcs'
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
        slurm_name="dual_td_nce",
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
        for env_name in ["online-ant-xy-v0"]:
            for tanh_squash in [True, False]:
                for const_std in [False]:
                    for normalize_q_loss in [True, False]:
                        for seed in [10, 20, 30]:
                            exp_name = f"{datetime.today().strftime('%Y%m%d')}_dual_td_nce_{env_name}_tanh_squash={tanh_squash}_const_std={const_std}_norm_q_loss={normalize_q_loss}"
                            log_dir = os.path.expanduser(
                                f"{log_root_dir}/exp_logs/dual_rl_logs/dual_td_nce/{exp_name}/{seed}")
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
                                python $PROJECT_DIR/main_online.py \
                                    --enable_wandb=1 \
                                    --env_name={env_name} \
                                    --train_steps=1_000_000 \
                                    --log_interval=5_000 \
                                    --eval_interval=100_000 \
                                    --save_interval=1_000_000 \
                                    --eval_episodes=50 \
                                    --agent=agents/dual_td_nce.py \
                                    --agent.tanh_squash={tanh_squash} \
                                    --agent.const_std={const_std} \
                                    --agent.normalize_q_loss={normalize_q_loss} \
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
