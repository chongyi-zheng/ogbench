rsync -av --progress --exclude='*.pkl' \
      -e ssh \
      'cz8792@neuronic.cs.princeton.edu:/n/fs/prl-chongyiz/exp_logs/ogbench_logs/hilp_fom_offline2offline/20250512_hilp_fom_offline2offline*' \
      /n/fs/rl-chongyiz/exp_logs/ogbench_logs/hilp_fom_offline2offline/
