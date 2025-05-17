rsync -av --progress --exclude='*.pkl' \
      -e ssh \
      'cz8792@neuronic.cs.princeton.edu:/n/fs/prl-chongyiz/exp_logs/ogbench_logs/fb_repr_offline2offline/20250512_fb_repr_offline2offline_*' \
      /n/fs/rl-chongyiz/exp_logs/ogbench_logs/fb_repr_offline2offline/
