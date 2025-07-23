rsync -av --progress --exclude='*.pkl' \
      -e ssh \
      'cz8792@cycles.cs.princeton.edu:/n/fs/rl-chongyiz/exp_logs/fdrl_logs/fdrl/20250718_fdrl_humanoidmaze-medium-navigate-singletask*' \
      /n/fs/prl-chongyiz/exp_logs/fdrl_logs/fdrl/


rsync -av --progress --exclude='*.pkl' \
      -e ssh \
      'cz8792@della9.princeton.edu:/home/cz8792/research/exorl/datasets/jaco/rnd/video/episode_017201.mp4' \
      /Users/chongyiz/Downloads/exorl_videos/jaco.mp4