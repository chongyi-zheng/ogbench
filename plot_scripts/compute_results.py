#!/usr/bin/env python3
"""
Compute mean ± std of the last k evaluation returns across many seeds.

Example:
    python aggregate_returns.py --root /path/to/parent --last 3
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    def str_pair(s):
        splited_s = s.split(',')
        assert splited_s, 'invalid string pair'
        return splited_s[0], splited_s[1]


    def str_triplet(s):
        splited_s = s.split(',')
        assert splited_s, 'invalid string pair'
        return splited_s[0], splited_s[1], splited_s[2]
    
    p = argparse.ArgumentParser(description="Aggregate evaluation/episode.return statistics.")
    p.add_argument(
        "--log_dir",
        type=str,
        default="/n/fs/rl-chongyiz/exp_logs/ogbench_logs"
    )
    p.add_argument(
        "--algos",
        nargs="+",
        type=str_triplet,
        default=[
            # # cheetah_run
            # ("sarsa_ifql_offline2offline", "cheetah_run", "20250416_sarsa_ifql_offline2offline_cheetah_run_obs_norm=normal_lr=0.0003_tau=0.005_alpha=0.3_num_fg=16_actor_freq=4_expectile=0.75_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.5_reward=state_bc_pretrain"), 
            # ("iql_offline2offline", "cheetah_run", "20250405_iql_offline2offline_cheetah_run_obs_norm_type=normal_alpha=1.0_actor_freq=4"), 
            # ("rebrac_offline2offline", "cheetah_run", "20250416_rebrac_offline2offline_cheetah_run_obs_norm_type=normal_alpha_actor=0.1_alpha_critic=0.1_actor_freq=4"), 
            # ("crl_infonce_offline2offline", "cheetah_run", "20250416_crl_infonce_offline2offline_cheetah_run_obs_norm_type=normal_alpha=0.03_reward_type=state_actor_freq=4"),
            # ("td_infonce_offline2offline", "cheetah_run", "20250418_td_infonce_offline2offline_cheetah_run_obs_norm_type=normal_alpha=0.003_reward_type=state_actor_freq=4"),
            # # cheetah_run_backward
            # ("sarsa_ifql_offline2offline", "cheetah_run_backward", "20250416_sarsa_ifql_offline2offline_cheetah_run_backward_obs_norm=normal_lr=0.0003_tau=0.005_alpha=0.3_num_fg=16_actor_freq=4_expectile=0.75_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.25_reward=state_bc_pretrain"), 
            # ("iql_offline2offline", "cheetah_run_backward", "20250416_iql_offline2offline_cheetah_run_backward_obs_norm_type=normal_alpha=10.0_expectile=0.99_actor_freq=4"), 
            # ("rebrac_offline2offline", "cheetah_run_backward", "20250416_rebrac_offline2offline_cheetah_run_backward_obs_norm_type=normal_alpha_actor=0.1_alpha_critic=0.1_actor_freq=4"), 
            # ("crl_infonce_offline2offline", "cheetah_run_backward", "20250416_crl_infonce_offline2offline_cheetah_run_backward_obs_norm_type=normal_alpha=0.003_reward_type=state_actor_freq=4"), 
            # ("td_infonce_offline2offline", "cheetah_run_backward", "20250418_td_infonce_offline2offline_cheetah_run_backward_obs_norm_type=normal_alpha=0.003_reward_type=state_actor_freq=4"), 
            # # cheetah_walk
            # ("sarsa_ifql_offline2offline", "cheetah_walk", "20250416_sarsa_ifql_offline2offline_cheetah_walk_obs_norm=normal_lr=0.0003_tau=0.005_alpha=0.3_num_fg=16_actor_freq=4_expectile=0.75_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.25_reward=state_bc_pretrain"), 
            # ("iql_offline2offline", "cheetah_walk", "20250416_iql_offline2offline_cheetah_walk_obs_norm_type=normal_alpha=10.0_expectile=0.99_actor_freq=4"), 
            # ("rebrac_offline2offline", "cheetah_walk", "20250416_rebrac_offline2offline_cheetah_walk_obs_norm_type=normal_alpha_actor=0.1_alpha_critic=0.1_actor_freq=4"), 
            # ("crl_infonce_offline2offline", "cheetah_walk", "20250416_crl_infonce_offline2offline_cheetah_walk_obs_norm_type=normal_alpha=0.003_reward_type=state_actor_freq=4"), 
            # ("td_infonce_offline2offline", "cheetah_walk", "20250418_td_infonce_offline2offline_cheetah_walk_obs_norm_type=normal_alpha=0.003_reward_type=state_actor_freq=4"), 
            # # cheetah_walk_backward
            # ("sarsa_ifql_offline2offline", "cheetah_walk_backward", "20250416_sarsa_ifql_offline2offline_cheetah_walk_backward_obs_norm=normal_lr=0.0003_tau=0.005_alpha=0.3_num_fg=16_actor_freq=4_expectile=0.75_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.05_reward=state_bc_pretrain"), 
            # ("iql_offline2offline", "cheetah_walk_backward", "20250416_iql_offline2offline_cheetah_walk_backward_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"), 
            # ("rebrac_offline2offline", "cheetah_walk_backward", "20250416_rebrac_offline2offline_cheetah_walk_backward_obs_norm_type=normal_alpha_actor=0.1_alpha_critic=0.1_actor_freq=4"),
            # ("crl_infonce_offline2offline", "cheetah_walk_backward", "20250416_crl_infonce_offline2offline_cheetah_walk_backward_obs_norm_type=normal_alpha=0.003_reward_type=state_actor_freq=4"), 
            # ("td_infonce_offline2offline", "cheetah_walk_backward", "20250418_td_infonce_offline2offline_cheetah_walk_backward_obs_norm_type=normal_alpha=0.003_reward_type=state_actor_freq=4"), 
            # # walker_walk
            # ("sarsa_ifql_offline2offline", "walker_walk", "20250418_sarsa_ifql_offline2offline_walker_walk_obs_norm=normal_lr=0.0003_tau=0.005_alpha=0.3_num_fg=16_actor_freq=4_expectile=0.85_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.25_reward=state_bc_pretrain"),
            # ("iql_offline2offline", "walker_walk", "20250418_iql_offline2offline_walker_walk_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"), 
            # ("rebrac_offline2offline", "walker_walk", "20250418_rebrac_offline2offline_walker_walk_obs_norm_type=normal_alpha_actor=0.1_alpha_critic=10.0_actor_freq=4"), 
            # ("crl_infonce_offline2offline", "walker_walk", "20250418_crl_infonce_offline2offline_walker_walk_obs_norm_type=normal_alpha=0.003_reward_type=state_actor_freq=4"), 
            # ("td_infonce_offline2offline", "walker_walk", "20250418_td_infonce_offline2offline_walker_walk_obs_norm_type=normal_alpha=0.003_reward_type=state_actor_freq=4"), 
            # # walker_run
            # ("sarsa_ifql_offline2offline", "walker_run", "20250418_sarsa_ifql_offline2offline_walker_run_obs_norm=normal_lr=0.0003_tau=0.005_alpha=0.3_num_fg=16_actor_freq=4_expectile=0.85_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.5_reward=state_bc_pretrain"),
            # ("iql_offline2offline", "walker_run", "20250418_iql_offline2offline_walker_run_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"), 
            # ("rebrac_offline2offline", "walker_run", "20250418_rebrac_offline2offline_walker_run_obs_norm_type=normal_alpha_actor=10.0_alpha_critic=0.1_actor_freq=4"), 
            # ("crl_infonce_offline2offline", "walker_run", "20250418_crl_infonce_offline2offline_walker_run_obs_norm_type=normal_alpha=0.003_reward_type=state_actor_freq=4"), 
            # ("td_infonce_offline2offline", "walker_run", "20250418_td_infonce_offline2offline_walker_run_obs_norm_type=normal_alpha=0.003_reward_type=state_actor_freq=4"), 
            # # walker_stand
            # ("sarsa_ifql_offline2offline", "walker_stand", "20250418_sarsa_ifql_offline2offline_walker_stand_obs_norm=normal_lr=0.0003_tau=0.005_alpha=0.3_num_fg=16_actor_freq=4_expectile=0.85_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.25_reward=state_bc_pretrain"),
            # ("iql_offline2offline", "walker_stand", "20250418_iql_offline2offline_walker_stand_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"), 
            # ("rebrac_offline2offline", "walker_stand", "20250418_rebrac_offline2offline_walker_stand_obs_norm_type=normal_alpha_actor=0.1_alpha_critic=0.1_actor_freq=4"), 
            # ("crl_infonce_offline2offline", "walker_stand", "20250418_crl_infonce_offline2offline_walker_stand_obs_norm_type=normal_alpha=0.03_reward_type=state_actor_freq=4"), 
            # ("td_infonce_offline2offline", "walker_stand", "20250418_td_infonce_offline2offline_walker_stand_obs_norm_type=normal_alpha=0.003_reward_type=state_actor_freq=4"), 
            # # walker_flip
            # ("sarsa_ifql_offline2offline", "walker_flip", "20250418_sarsa_ifql_offline2offline_walker_flip_obs_norm=normal_lr=0.0003_tau=0.005_alpha=0.3_num_fg=16_actor_freq=4_expectile=0.85_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.5_reward=state_bc_pretrain"),
            # ("iql_offline2offline", "walker_flip", "20250418_iql_offline2offline_walker_flip_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"),
            # ("rebrac_offline2offline", "walker_flip", "20250418_rebrac_offline2offline_walker_flip_obs_norm_type=normal_alpha_actor=0.1_alpha_critic=0.1_actor_freq=4"),
            # ("crl_infonce_offline2offline", "walker_flip", "20250418_crl_infonce_offline2offline_walker_flip_obs_norm_type=normal_alpha=0.03_reward_type=state_actor_freq=4"), 
            # ("td_infonce_offline2offline", "walker_flip", "20250418_td_infonce_offline2offline_walker_flip_obs_norm_type=normal_alpha=0.03_reward_type=state_actor_freq=4"), 
            # # quadruped_run
            # ("sarsa_ifql_offline2offline", "quadruped_run", "20250420_sarsa_ifql_offline2offline_quadruped_run_obs_norm=normal_lr=0.0003_tau=0.005_alpha=3.0_num_fg=16_actor_freq=4_expectile=0.75_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.05_reward=state_bc_pretrain"),
            # ("iql_offline2offline", "quadruped_run", "20250420_iql_offline2offline_quadruped_run_obs_norm_type=normal_alpha=10.0_expectile=0.99_actor_freq=4"),
            # ("rebrac_offline2offline", "quadruped_run", "20250420_rebrac_offline2offline_quadruped_run_obs_norm_type=normal_alpha_actor=1.0_alpha_critic=1.0_actor_freq=4"),
            # ("crl_infonce_offline2offline", "quadruped_run", "20250420_crl_infonce_offline2offline_quadruped_run_obs_norm_type=normal_alpha=0.03_reward_type=state_actor_freq=4"),
            # ("td_infonce_offline2offline", "quadruped_run", "20250420_td_infonce_offline2offline_quadruped_run_obs_norm_type=normal_alpha=0.03_reward_type=state_actor_freq=4"),
            # # quadruped_jump
            # ("sarsa_ifql_offline2offline", "quadruped_jump", "20250420_sarsa_ifql_offline2offline_quadruped_jump_obs_norm=normal_lr=0.0003_tau=0.005_alpha=3.0_num_fg=16_actor_freq=4_expectile=0.75_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.1_reward=state_bc_pretrain"),
            # ("iql_offline2offline", "quadruped_jump", "20250420_iql_offline2offline_quadruped_jump_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"),
            # ("rebrac_offline2offline", "quadruped_jump", "20250420_rebrac_offline2offline_quadruped_jump_obs_norm_type=normal_alpha_actor=1.0_alpha_critic=1.0_actor_freq=4"),
            # ("crl_infonce_offline2offline", "quadruped_jump", "20250420_crl_infonce_offline2offline_quadruped_jump_obs_norm_type=normal_alpha=0.03_reward_type=state_actor_freq=4"),
            # ("td_infonce_offline2offline", "quadruped_jump", "20250420_td_infonce_offline2offline_quadruped_jump_obs_norm_type=normal_alpha=0.03_reward_type=state_actor_freq=4"),
            # # quadruped_stand
            # ("sarsa_ifql_offline2offline", "quadruped_stand", "20250420_sarsa_ifql_offline2offline_quadruped_stand_obs_norm=normal_lr=0.0003_tau=0.005_alpha=3.0_num_fg=16_actor_freq=4_expectile=0.75_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.1_reward=state_bc_pretrain"),
            # ("iql_offline2offline", "quadruped_stand", "20250420_iql_offline2offline_quadruped_stand_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"),
            # ("rebrac_offline2offline", "quadruped_stand", "20250420_rebrac_offline2offline_quadruped_stand_obs_norm_type=normal_alpha_actor=1.0_alpha_critic=1.0_actor_freq=4"),
            # ("crl_infonce_offline2offline", "quadruped_stand", "20250420_crl_infonce_offline2offline_quadruped_stand_obs_norm_type=normal_alpha=0.03_reward_type=state_actor_freq=4"),
            # ("td_infonce_offline2offline", "quadruped_stand", "20250420_td_infonce_offline2offline_quadruped_stand_obs_norm_type=normal_alpha=0.03_reward_type=state_actor_freq=4"),
            # # quadruped_walk
            # ("sarsa_ifql_offline2offline", "quadruped_walk", "20250420_sarsa_ifql_offline2offline_quadruped_walk_obs_norm=normal_lr=0.0003_tau=0.005_alpha=3.0_num_fg=16_actor_freq=4_expectile=0.75_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.05_reward=state_bc_pretrain"),
            # ("iql_offline2offline", "quadruped_walk", "20250420_iql_offline2offline_quadruped_walk_obs_norm_type=normal_alpha=10.0_expectile=0.99_actor_freq=4"),
            # ("rebrac_offline2offline", "quadruped_walk", "20250420_rebrac_offline2offline_quadruped_walk_obs_norm_type=normal_alpha_actor=1.0_alpha_critic=1.0_actor_freq=4"),
            # ("crl_infonce_offline2offline", "quadruped_walk", "20250420_crl_infonce_offline2offline_quadruped_walk_obs_norm_type=normal_alpha=0.03_reward_type=state_actor_freq=4"),
            # ("td_infonce_offline2offline", "quadruped_walk", "20250420_td_infonce_offline2offline_quadruped_walk_obs_norm_type=normal_alpha=0.03_reward_type=state_actor_freq=4"),
            # # jaco_reach_top_left
            # ("sarsa_ifql_offline2offline", "jaco_reach_top_left", "20250420_sarsa_ifql_offline2offline_jaco_reach_top_left_obs_norm=normal_lr=0.0003_tau=0.005_alpha=0.1_num_fg=16_actor_freq=4_expectile=0.75_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=1.0_reward=state_bc_pretrain"),
            # ("iql_offline2offline", "jaco_reach_top_left", "20250420_iql_offline2offline_jaco_reach_top_left_obs_norm_type=normal_alpha=10.0_expectile=0.99_actor_freq=4"),
            # ("rebrac_offline2offline", "jaco_reach_top_left", "20250420_rebrac_offline2offline_jaco_reach_top_left_obs_norm_type=normal_alpha_actor=0.1_alpha_critic=0.1_actor_freq=4"),
            # ("crl_infonce_offline2offline", "jaco_reach_top_left", "20250420_crl_infonce_offline2offline_jaco_reach_top_left_obs_norm_type=normal_alpha=0.003_reward_type=state_actor_freq=4"),
            # ("td_infonce_offline2offline", "jaco_reach_top_left", "20250420_td_infonce_offline2offline_jaco_reach_top_left_obs_norm_type=normal_alpha=0.03_reward_type=state_actor_freq=4"),
            # # jaco_reach_top_right
            # ("sarsa_ifql_offline2offline", "jaco_reach_top_right", "20250420_sarsa_ifql_offline2offline_jaco_reach_top_right_obs_norm=normal_lr=0.0003_tau=0.005_alpha=0.1_num_fg=16_actor_freq=4_expectile=0.75_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=1.0_reward=state_bc_pretrain"),
            # ("iql_offline2offline", "jaco_reach_top_right", "20250420_iql_offline2offline_jaco_reach_top_right_obs_norm_type=normal_alpha=10.0_expectile=0.99_actor_freq=4"),
            # ("rebrac_offline2offline", "jaco_reach_top_right", "20250420_rebrac_offline2offline_jaco_reach_top_right_obs_norm_type=normal_alpha_actor=0.1_alpha_critic=0.1_actor_freq=4"),
            # ("crl_infonce_offline2offline", "jaco_reach_top_right", "20250420_crl_infonce_offline2offline_jaco_reach_top_right_obs_norm_type=normal_alpha=0.003_reward_type=state_actor_freq=4"),
            # ("td_infonce_offline2offline", "jaco_reach_top_right", "20250420_td_infonce_offline2offline_jaco_reach_top_right_obs_norm_type=normal_alpha=0.03_reward_type=state_actor_freq=4"),
            # # jaco_reach_bottom_left
            # ("sarsa_ifql_offline2offline", "jaco_reach_bottom_left", "20250420_sarsa_ifql_offline2offline_jaco_reach_bottom_left_obs_norm=normal_lr=0.0003_tau=0.005_alpha=0.1_num_fg=16_actor_freq=4_expectile=0.75_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=1.0_reward=state_bc_pretrain"),
            # ("iql_offline2offline", "jaco_reach_bottom_left", "20250420_iql_offline2offline_jaco_reach_bottom_right_obs_norm_type=normal_alpha=10.0_expectile=0.99_actor_freq=4"),
            # ("rebrac_offline2offline", "jaco_reach_bottom_left", "20250420_rebrac_offline2offline_jaco_reach_bottom_left_obs_norm_type=normal_alpha_actor=0.1_alpha_critic=0.1_actor_freq=4"),
            # ("crl_infonce_offline2offline", "jaco_reach_bottom_left", "20250420_crl_infonce_offline2offline_jaco_reach_bottom_left_obs_norm_type=normal_alpha=0.003_reward_type=state_actor_freq=4"),
            # ("td_infonce_offline2offline", "jaco_reach_bottom_left", "20250420_td_infonce_offline2offline_jaco_reach_bottom_left_obs_norm_type=normal_alpha=0.03_reward_type=state_actor_freq=4"),
            # # jaco_reach_bottom_right
            # ("sarsa_ifql_offline2offline", "jaco_reach_bottom_right", "20250420_sarsa_ifql_offline2offline_jaco_reach_bottom_right_obs_norm=normal_lr=0.0003_tau=0.005_alpha=0.1_num_fg=16_actor_freq=4_expectile=0.75_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=1.0_reward=state_bc_pretrain"),
            # ("iql_offline2offline", "jaco_reach_bottom_right", "20250420_iql_offline2offline_jaco_reach_bottom_left_obs_norm_type=normal_alpha=10.0_expectile=0.99_actor_freq=4"),
            # ("rebrac_offline2offline", "jaco_reach_bottom_right", "20250420_rebrac_offline2offline_jaco_reach_bottom_right_obs_norm_type=normal_alpha_actor=0.1_alpha_critic=0.1_actor_freq=4"),
            # ("crl_infonce_offline2offline", "jaco_reach_bottom_right", "20250420_crl_infonce_offline2offline_jaco_reach_bottom_right_obs_norm_type=normal_alpha=0.003_reward_type=state_actor_freq=4"),
            # ("td_infonce_offline2offline", "jaco_reach_bottom_right", "20250420_td_infonce_offline2offline_jaco_reach_bottom_right_obs_norm_type=normal_alpha=0.03_reward_type=state_actor_freq=4"),

            # cube-single-play-singletask-task1-v0 
            ("sarsa_ifql_offline2offline", "cube-single-play-singletask-task1-v0", "20250419_sarsa_ifql_offline2offline_cube-single-play-singletask-task1-v0_obs_norm=normal_lr=0.0003_tau=0.005_alpha=30.0_num_fg=16_actor_freq=4_expectile=0.9_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.25_reward=state_bc_pretrain"),
            ("iql_offline2offline", "cube-single-play-singletask-task1-v0", "20250420_iql_offline2offline_cube-single-play-singletask-task1-v0_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"),
            ("rebrac_offline2offline", "cube-single-play-singletask-task1-v0", "20250420_rebrac_offline2offline_cube-single-play-singletask-task1-v0_obs_norm_type=normal_alpha_actor=1.0_alpha_critic=1.0_actor_freq=4"),
            # cube-single-play-singletask-task2-v0 
            ("sarsa_ifql_offline2offline", "cube-single-play-singletask-task2-v0", "20250419_sarsa_ifql_offline2offline_cube-single-play-singletask-task2-v0_obs_norm=normal_lr=0.0003_tau=0.005_alpha=30.0_num_fg=16_actor_freq=4_expectile=0.9_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.25_reward=state_bc_pretrain"),
            ("iql_offline2offline", "cube-single-play-singletask-task2-v0", "20250420_iql_offline2offline_cube-single-play-singletask-task2-v0_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"),
            ("rebrac_offline2offline", "cube-single-play-singletask-task2-v0", "20250420_rebrac_offline2offline_cube-single-play-singletask-task2-v0_obs_norm_type=normal_alpha_actor=1.0_alpha_critic=1.0_actor_freq=4"),
            # cube-single-play-singletask-task3-v0 
            ("sarsa_ifql_offline2offline", "cube-single-play-singletask-task3-v0", "20250419_sarsa_ifql_offline2offline_cube-single-play-singletask-task3-v0_obs_norm=normal_lr=0.0003_tau=0.005_alpha=30.0_num_fg=16_actor_freq=4_expectile=0.9_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.25_reward=state_bc_pretrain"),
            ("iql_offline2offline", "cube-single-play-singletask-task3-v0", "20250420_iql_offline2offline_cube-single-play-singletask-task3-v0_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"),
            ("rebrac_offline2offline", "cube-single-play-singletask-task3-v0", "20250420_rebrac_offline2offline_cube-single-play-singletask-task3-v0_obs_norm_type=normal_alpha_actor=1.0_alpha_critic=1.0_actor_freq=4"),
            # cube-single-play-singletask-task4-v0 
            ("sarsa_ifql_offline2offline", "cube-single-play-singletask-task4-v0", "20250419_sarsa_ifql_offline2offline_cube-single-play-singletask-task4-v0_obs_norm=normal_lr=0.0003_tau=0.005_alpha=30.0_num_fg=16_actor_freq=4_expectile=0.9_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.25_reward=state_bc_pretrain"),
            ("iql_offline2offline", "cube-single-play-singletask-task4-v0", "20250420_iql_offline2offline_cube-single-play-singletask-task4-v0_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"),
            ("rebrac_offline2offline", "cube-single-play-singletask-task4-v0", "20250420_rebrac_offline2offline_cube-single-play-singletask-task4-v0_obs_norm_type=normal_alpha_actor=1.0_alpha_critic=1.0_actor_freq=4"),
            # cube-single-play-singletask-task5-v0 
            ("sarsa_ifql_offline2offline", "cube-single-play-singletask-task5-v0", "20250419_sarsa_ifql_offline2offline_cube-single-play-singletask-task5-v0_obs_norm=normal_lr=0.0003_tau=0.005_alpha=30.0_num_fg=16_actor_freq=4_expectile=0.9_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.25_reward=state_bc_pretrain"),
            ("iql_offline2offline", "cube-single-play-singletask-task5-v0", "20250420_iql_offline2offline_cube-single-play-singletask-task5-v0_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"),
            ("rebrac_offline2offline", "cube-single-play-singletask-task5-v0", "20250420_rebrac_offline2offline_cube-single-play-singletask-task5-v0_obs_norm_type=normal_alpha_actor=1.0_alpha_critic=1.0_actor_freq=4"),
            # cube-double-play-singletask-task1-v0
            ("sarsa_ifql_offline2offline", "cube-single-play-singletask-task1-v0", "20250419_sarsa_ifql_offline2offline_cube-double-play-singletask-task1-v0_obs_norm=normal_lr=0.0003_tau=0.005_alpha=30.0_num_fg=16_actor_freq=4_expectile=0.9_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.05_reward=state_bc_pretrain"),
            ("iql_offline2offline", "cube-single-play-singletask-task1-v0", "20250420_iql_offline2offline_cube-double-play-singletask-task1-v0_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"),
            ("rebrac_offline2offline", "cube-single-play-singletask-task1-v0", "20250420_rebrac_offline2offline_cube-double-play-singletask-task1-v0_obs_norm_type=normal_alpha_actor=1.0_alpha_critic=1.0_actor_freq=4"),
            # cube-double-play-singletask-task2-v0
            ("sarsa_ifql_offline2offline", "cube-double-play-singletask-task2-v0", "20250419_sarsa_ifql_offline2offline_cube-double-play-singletask-task2-v0_obs_norm=normal_lr=0.0003_tau=0.005_alpha=30.0_num_fg=16_actor_freq=4_expectile=0.9_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.05_reward=state_bc_pretrain"),
            ("iql_offline2offline", "cube-double-play-singletask-task2-v0", "20250420_iql_offline2offline_cube-double-play-singletask-task2-v0_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"),
            ("rebrac_offline2offline", "cube-double-play-singletask-task2-v0", "20250420_rebrac_offline2offline_cube-double-play-singletask-task2-v0_obs_norm_type=normal_alpha_actor=1.0_alpha_critic=1.0_actor_freq=4"),
            # cube-double-play-singletask-task3-v0
            ("sarsa_ifql_offline2offline", "cube-single-play-singletask-task3-v0", "20250419_sarsa_ifql_offline2offline_cube-double-play-singletask-task3-v0_obs_norm=normal_lr=0.0003_tau=0.005_alpha=30.0_num_fg=16_actor_freq=4_expectile=0.9_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.05_reward=state_bc_pretrain"),
            ("iql_offline2offline", "cube-single-play-singletask-task3-v0", "20250420_iql_offline2offline_cube-double-play-singletask-task3-v0_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"),
            ("rebrac_offline2offline", "cube-single-play-singletask-task3-v0", "20250420_rebrac_offline2offline_cube-double-play-singletask-task3-v0_obs_norm_type=normal_alpha_actor=1.0_alpha_critic=1.0_actor_freq=4"),
            # cube-double-play-singletask-task4-v0
            ("sarsa_ifql_offline2offline", "cube-single-play-singletask-task4-v0", "20250419_sarsa_ifql_offline2offline_cube-double-play-singletask-task4-v0_obs_norm=normal_lr=0.0003_tau=0.005_alpha=30.0_num_fg=16_actor_freq=4_expectile=0.9_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.05_reward=state_bc_pretrain"),
            ("iql_offline2offline", "cube-single-play-singletask-task4-v0", "20250420_iql_offline2offline_cube-double-play-singletask-task4-v0_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"),
            ("rebrac_offline2offline", "cube-single-play-singletask-task4-v0", "20250420_rebrac_offline2offline_cube-double-play-singletask-task4-v0_obs_norm_type=normal_alpha_actor=1.0_alpha_critic=1.0_actor_freq=4"),
            # cube-double-play-singletask-task5-v0
            ("sarsa_ifql_offline2offline", "cube-single-play-singletask-task5-v0", "20250419_sarsa_ifql_offline2offline_cube-double-play-singletask-task5-v0_obs_norm=normal_lr=0.0003_tau=0.005_alpha=30.0_num_fg=16_actor_freq=4_expectile=0.9_q_agg=min_clip_fgs=True_mixup=True_mixup_bw=0.05_reward=state_bc_pretrain"),
            ("iql_offline2offline", "cube-single-play-singletask-task5-v0", "20250420_iql_offline2offline_cube-double-play-singletask-task5-v0_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"),
            ("rebrac_offline2offline", "cube-single-play-singletask-task5-v0", "20250420_rebrac_offline2offline_cube-double-play-singletask-task5-v0_obs_norm_type=normal_alpha_actor=1.0_alpha_critic=1.0_actor_freq=4"),
            # scene-play-singletask-task1-v0
            ("sarsa_ifql_offline2offline", "scene-play-singletask-task1-v0", ""),
            ("iql_offline2offline", "scene-play-singletask-task1-v0", "20250420_iql_offline2offline_scene-play-singletask-task1-v0_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"),
            ("rebrac_offline2offline", "scene-play-singletask-task1-v0", "20250420_rebrac_offline2offline_scene-play-singletask-task1-v0_obs_norm_type=normal_alpha_actor=1.0_alpha_critic=1.0_actor_freq=4"),
            # scene-play-singletask-task2-v0
            ("sarsa_ifql_offline2offline", "scene-play-singletask-task2-v0", ""),
            ("iql_offline2offline", "scene-play-singletask-task2-v0", "20250420_iql_offline2offline_scene-play-singletask-task2-v0_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"),
            ("rebrac_offline2offline", "scene-play-singletask-task2-v0", "20250420_rebrac_offline2offline_scene-play-singletask-task2-v0_obs_norm_type=normal_alpha_actor=1.0_alpha_critic=1.0_actor_freq=4"),
            # scene-play-singletask-task3-v0
            ("sarsa_ifql_offline2offline", "scene-play-singletask-task3-v0", ""),
            ("iql_offline2offline", "scene-play-singletask-task3-v0", "20250420_iql_offline2offline_scene-play-singletask-task3-v0_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"),
            ("rebrac_offline2offline", "scene-play-singletask-task3-v0", "20250420_rebrac_offline2offline_scene-play-singletask-task3-v0_obs_norm_type=normal_alpha_actor=1.0_alpha_critic=1.0_actor_freq=4"),
            # scene-play-singletask-task4-v0
            ("sarsa_ifql_offline2offline", "scene-play-singletask-task4-v0", ""),
            ("iql_offline2offline", "scene-play-singletask-task4-v0", "20250420_iql_offline2offline_scene-play-singletask-task4-v0_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"),
            ("rebrac_offline2offline", "scene-play-singletask-task4-v0", "20250420_rebrac_offline2offline_scene-play-singletask-task4-v0_obs_norm_type=normal_alpha_actor=1.0_alpha_critic=1.0_actor_freq=4"),
            # scene-play-singletask-task5-v0
            ("sarsa_ifql_offline2offline", "scene-play-singletask-task5-v0", ""),
            ("iql_offline2offline", "scene-play-singletask-task5-v0", "20250420_iql_offline2offline_scene-play-singletask-task5-v0_obs_norm_type=normal_alpha=1.0_expectile=0.99_actor_freq=4"),
            ("rebrac_offline2offline", "scene-play-singletask-task5-v0", "20250420_rebrac_offline2offline_scene-play-singletask-task5-v0_obs_norm_type=normal_alpha_actor=1.0_alpha_critic=1.0_actor_freq=4"),
            # puzzle-4x4-play-singletask-task1-v0
        ]
    )
    p.add_argument(
        "--filename",
        type=str,
        default="finetuning_eval.csv",
    )
    p.add_argument(
        "--stat_name",
        type=str,
        # default="evaluation/episode.return",
        default="evaluation/episode.success",
    )
    p.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[1_400_000, 1_450_000, 1_500_000],
    )
    return p.parse_args()


def find_csv_files(log_dir, algos, filename) -> list[Path]:
    # Matches …/<SEED>/debug/**/finetuning_eval.csv  (depth under debug doesn't matter)
    
    csv_files = dict()
    for algo, env_name, exp_log_dir in algos:
        pattern = os.path.join(log_dir, algo, exp_log_dir, "*", "debug", "**", filename)
        files = [p for p in glob.glob(pattern, recursive=True)]
        if env_name not in csv_files:
            csv_files[env_name] = {algo: files}
        else:
            csv_files[env_name][algo] = files 
    return csv_files


def load_data(csv_path, stat_name, steps) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if stat_name not in df.columns:
        raise KeyError(f"{csv_path} doesn't contain 'evaluation/episode.return'")
    data = df.loc[df["step"].isin(steps), stat_name].values
    return data


def main():
    args = parse_args()
    csv_files = find_csv_files(args.log_dir, args.algos, args.filename)
    if not csv_files:
        print("No finetuning_eval.csv files found. Check the --root path.")
        return

    for algo, algo_csv_files in csv_files.items():
        for env_name, csv_files in algo_csv_files.items():
            seed_data = []
            for csv_file in csv_files:
                data = load_data(csv_file, args.stat_name, args.steps)
                seed_data.append(data)

            seed_data = np.asarray(seed_data)
            seed_data = np.mean(seed_data, axis=-1)
            if len(seed_data) == 1:
                print("Warning: only one random seed!")
            mean = np.mean(seed_data)
            std = np.std(seed_data, ddof=1)  # sample std (N ‑ 1 in the denominator)

            print(f"env = {algo}, {env_name}, steps = {args.steps}: mean = {mean:.4f}, std = {std:.4f}")

    # if not all_means:
    #     print("No usable data found.")
    #     return

    # overall_mean = np.mean(all_means)
    # overall_std = np.std(all_means, ddof=1)  # sample std (N‑1 in denominator)

    # print("\n" + "-" * 60)
    # print(
    #     f"Across {len(all_means)} seeds: "
    #     f"mean = {overall_mean:.3f},  std = {overall_std:.3f}  (based on each seed’s mean of last {args.last})"
    # )


if __name__ == "__main__":
    main()
