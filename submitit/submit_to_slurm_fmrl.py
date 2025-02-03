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
        slurm_name="fmrl",
        slurm_time=int(8 * 60),  # minute
        slurm_partition=partition,
        slurm_account=account,
        slurm_nodes=1,
        slurm_ntasks_per_node=1,  # tasks can share nodes
        slurm_cpus_per_task=8,
        slurm_mem="16G",
        slurm_gpus_per_node=1,
        slurm_stderr_to_stdout=True,
    )

    # sfbc hyperparameters: eval_temperature, eval_task_id, const_std, num_flow_steps, num_candidates, distill_type
    # awr hyperparameters: eval_temperature, eval_task_id, alpha, const_std, num_flow_steps, distill_type
    # ddpgbc hyperparameters: eval_temperature, eval_task_id, alpha, const_std, num_flow_steps, num_candidates, distill_type
    # pgbc hyperparameters: eval_temperature, eval_task_id, alpha, const_std, num_flow_steps, distill_type
    with executor.batch():  # job array
        for env_name in [
            "antmaze-large-navigate-singletask-v0",
            # "humanoidmaze-medium-navigate-singletask-v0",
            # "antsoccer-arena-navigate-singletask-v0"
        ]:
            for discount in [0.99]:
                for alpha in [0.03, 0.3, 3, 30, 300]:
                    for const_std in [True, False]:
                        for num_flow_steps in [10]:
                            for q_agg in ["min"]:
                                for critic_layer_norm in [False]:
                                    for distill_type in ['none', 'fwd_int']:
                                        for seed in [0, 1]:
                                            exp_name = f"{datetime.today().strftime('%Y%m%d')}_fmrl_{env_name}_discount={discount}_alpha={alpha}_const_std={const_std}_num_flow_steps={num_flow_steps}_q_agg={q_agg}_critic_layer_norm={critic_layer_norm}_distill_type={distill_type}_normalized_q"
                                            log_dir = os.path.expanduser(
                                                f"{log_root_dir}/exp_logs/ogbench_logs/fmrl/{exp_name}/{seed}")

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
                                                export WANDB_API_KEY=bbb3bca410f71c2d7cfe6fe0bbe55a38d1015831;

                                                rm -rf {log_dir};
                                                mkdir -p {log_dir};
                                                python $PROJECT_DIR/impls/main_rl.py \
                                                    --enable_wandb=1 \
                                                    --env_name={env_name} \
                                                    --eval_episodes=50 \
                                                    --dataset_class=GCDataset \
                                                    --agent=impls/agents/fmrl.py \
                                                    --agent.discount={discount} \
                                                    --agent.alpha={alpha} \
                                                    --agent.const_std={const_std} \
                                                    --agent.num_flow_steps={num_flow_steps} \
                                                    --agent.q_agg={q_agg} \
                                                    --agent.critic_layer_norm={critic_layer_norm} \
                                                    --agent.distill_type={distill_type} \
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
