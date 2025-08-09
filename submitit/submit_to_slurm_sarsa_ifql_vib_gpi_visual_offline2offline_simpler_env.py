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
    elif 'della' in cluster_name:
        log_root_dir = '/home/cz8792/gpfs'
        # partition = 'gpu-test'
        # account = None
        partition = 'pli'
        account = 'rlchongyiz'
        exclude = 'della-j16g3,della-k12g2,della-k14g1'
    elif cluster_name in ['soak.cs.princeton.edu', 'wash.cs.princeton.edu',
                          'rinse.cs.princeton.edu', 'spin.cs.princeton.edu']:
        log_root_dir = '/n/fs/rl-chongyiz'
        partition = None
        account = 'pnlp'
        exclude = None
    elif cluster_name == 'neuronic.cs.princeton.edu':
        log_root_dir = '/n/fs/prl-chongyiz'
        partition = 'all'
        account = None
        exclude = None
    else:
        raise NotImplementedError

    executor = submitit.AutoExecutor(folder="/tmp/submitit_logs")  # this path is not actually used.
    executor.update_parameters(
        slurm_name="sarsa_ifql_vib_gpi_offline2offline",
        slurm_time=int(24 * 60),  # minute
        slurm_partition=partition,
        slurm_account=account,
        slurm_nodes=1,
        slurm_ntasks_per_node=1,  # tasks can share nodes
        slurm_cpus_per_task=16,
        slurm_mem="400G",
        slurm_gpus_per_node=1,
        slurm_exclude=exclude,
        slurm_stderr_to_stdout=True,
        slurm_array_parallelism=10,
    )

    with executor.batch():  # job array
        for env_name in [
            "google_robot_pick_coke_can"
        ]:
            for alpha in [30000, 3000, 300]:
                for num_flow_goals in [16]:
                    for actor_freq in [4]:
                        for expectile in [0.9]:
                            for kl_weight in [0.01, 0.005]:
                                for latent_dim in [512]:
                                    for value_layer_norm in [True]:
                                        for seed in [10, 20]:
                                            exp_name = f"{datetime.today().strftime('%Y%m%d')}_sarsa_ifql_vib_gpi_offline2offline_{env_name}_alpha={alpha}_num_fg={num_flow_goals}_actor_freq={actor_freq}_expectile={expectile}_kl_weight={kl_weight}_latent_dim={latent_dim}_value_ln={value_layer_norm}"
                                            log_dir = os.path.expanduser(
                                                f"{log_root_dir}/exp_logs/ogbench_logs/sarsa_ifql_vib_gpi_offline2offline/{exp_name}/{seed}")

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
                                                python $PROJECT_DIR/impls/main_simpler_env_offline2offline.py \
                                                    --enable_wandb=1 \
                                                    --env_name={env_name} \
                                                    --eval_episodes=50 \
                                                    --p_aug=0.5 \
                                                    --frame_stack=3 \
                                                    --finetuning_size=500_000 \
                                                    --pretraining_steps=250_000 \
                                                    --finetuning_steps=250_000 \
                                                    --eval_interval=25_000 \
                                                    --save_interval=500_000 \
                                                    --vqvae_restore_path /home/cz8792/gpfs/exp_logs/ogbench_logs/vqvae/20250731_vqvae_google_robot_pick_coke_can_quantizer_type=kl_frame_stack=3/10/debug/sd010_s_66455138.0.20250731_174433 \
                                                    --vqvae_restore_epoch 300_000 \
                                                    --agent=impls/agents/sarsa_ifql_vib_gpi.py \
                                                    --agent.batch_size=256 \
                                                    --agent.transition_hidden_dims="(512,512,512,512)" \
                                                    --agent.actor_hidden_dims="(512,512,512,512)" \
                                                    --agent.value_hidden_dims="(512,512,512,512)" \
                                                    --agent.reward_hidden_dims="(512,512,512,512)" \
                                                    --agent.lr=3e-4 \
                                                    --agent.tau=0.005 \
                                                    --agent.network_type=mlp \
                                                    --agent.num_residual_blocks=1 \
                                                    --agent.alpha={alpha} \
                                                    --agent.num_flow_steps=10 \
                                                    --agent.critic_noise_type=normal \
                                                    --agent.critic_fm_loss_type=sarsa_squared \
                                                    --agent.num_flow_goals={num_flow_goals} \
                                                    --agent.actor_freq={actor_freq} \
                                                    --agent.clip_flow_goals=False \
                                                    --agent.expectile={expectile} \
                                                    --agent.kl_weight={kl_weight} \
                                                    --agent.critic_latent_type=prior \
                                                    --agent.vector_field_time_sin_embedding=False \
                                                    --agent.latent_dim={latent_dim} \
                                                    --agent.q_agg=min \
                                                    --agent.transition_layer_norm=True \
                                                    --agent.reward_layer_norm=True \
                                                    --agent.actor_layer_norm=False \
                                                    --agent.value_layer_norm={value_layer_norm} \
                                                    --agent.normalize_q_loss=False \
                                                    --agent.use_mixup=False \
                                                    --agent.reward_type=state \
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
