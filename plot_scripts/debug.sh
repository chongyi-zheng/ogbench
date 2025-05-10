rsync -av --progress --exclude='*.pkl' \
      -e ssh \
      'cz8792@neuronic.cs.princeton.edu:/n/fs/prl-chongyiz/exp_logs/ogbench_logs/rebrac_offline2offline/20250510_rebrac_offline2offline_cheetah_run_*' \
      /n/fs/rl-chongyiz/exp_logs/ogbench_logs/rebrac_offline2offline/

