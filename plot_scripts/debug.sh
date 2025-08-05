rsync -av --progress \
      -e ssh \
      'cz8792@neuronic.cs.princeton.edu:/n/fs/prl-chongyiz/exp_logs/ogbench_logs/rebrac_offline2offline/20250804_*' \
      /n/fs/rl-chongyiz/exp_logs/ogbench_logs/rebrac_offline2offline/


rsync -av --progress --exclude='*.pkl' \
      -e ssh \
      'cz8792@della9.princeton.edu:/home/cz8792/research/exorl/datasets/jaco/rnd/video/episode_017201.mp4' \
      /Users/chongyiz/Downloads/exorl_videos/jaco.mp4

