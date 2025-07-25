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
        slurm_name="gcfac",
        slurm_time=int(8 * 60),  # minute
        slurm_partition=partition,
        slurm_account=account,
        slurm_nodes=1,
        slurm_ntasks_per_node=1,  # tasks can share nodes
        slurm_cpus_per_task=8,
        slurm_mem="16G",
        slurm_gpus_per_node=1,
        slurm_stderr_to_stdout=True,
        slurm_array_parallelism=24,
    )

    # ddpgbc hyperparameters: normalize_observation, alpha, const_std, num_flow_steps, exact_divergence, distill_type
    with executor.batch():  # job array
        for env_name in [
            # "pointmaze-medium-navigate-v0",
            # "pointmaze-large-navigate-v0",
            # "pointmaze-large-stitch-v0",
            "antmaze-large-navigate-v0",
            # "humanoidmaze-medium-navigate-v0",
            # "antsoccer-arena-navigate-v0"
            "cube-single-play-v0",
        ]:
            for obs_norm_type in ['normal']:
                for alpha in [0.3, 3.0, 30.0]:  # when normalize_q_loss = 1, use alpha around 0.003
                    for ode_solver_type in ['euler', 'dopri5']:
                        for ode_adjoint_type in ['recursive_checkpoint', 'direct']:
                            for div_type in ['hutchinson_rademacher']:  # both works similar
                                for hutchinson_prod_type in ['vjp']:
                                    for distill_type in ['log_prob', 'noise_div_int']:  # no distillation seems to work better
                                        for distill_loss_type in ['mse', 'expectile']:
                                            for actor_distill_type in ['fwd_sample']:
                                                for expectile in [0.85, 0.9, 0.95]:
                                                    for normalize_q_loss in [True]:  # it is important to normalize Q
                                                        for seed in [20]:
                                                            exp_name = f"{datetime.today().strftime('%Y%m%d')}_gcfac_env_name={env_name}_obs_norm={obs_norm_type}_alpha={alpha}_solver={ode_solver_type}_adjoint={ode_adjoint_type}_div={div_type}_hutchinson_prod={hutchinson_prod_type}_distill={distill_type}_distill_loss={distill_loss_type}_actor_distill={actor_distill_type}_expectile={expectile}_norm_q={normalize_q_loss}"
                                                            log_dir = os.path.expanduser(
                                                                f"{log_root_dir}/exp_logs/ogbench_logs/gcfac/{exp_name}/{seed}")

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
                                                                export EQX_ON_ERROR=nan;
                                                                export D4RL_SUPPRESS_IMPORT_ERROR=1;
                                                                export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia;
        
                                                                rm -rf {log_dir};
                                                                mkdir -p {log_dir};
                                                                python $PROJECT_DIR/impls/main.py \
                                                                    --enable_wandb=1 \
                                                                    --env_name={env_name} \
                                                                    --dataset_class=GCDataset \
                                                                    --obs_norm_type={obs_norm_type} \
                                                                    --eval_episodes=50 \
                                                                    --eval_on_cpu=0 \
                                                                    --eval_temperature=0.0 \
                                                                    --agent=impls/agents/gcfac.py \
                                                                    --agent.alpha={alpha} \
                                                                    --agent.ode_solver_type={ode_solver_type} \
                                                                    --agent.ode_adjoint_type={ode_adjoint_type} \
                                                                    --agent.num_flow_steps=10 \
                                                                    --agent.div_type={div_type} \
                                                                    --agent.distill_type={distill_type} \
                                                                    --agent.actor_distill_type={actor_distill_type} \
                                                                    --agent.use_target_network=False \
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
