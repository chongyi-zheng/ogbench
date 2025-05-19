rsync -av --progress --exclude='*.pkl' \
      -e ssh \
      'cz8792@neuronic.cs.princeton.edu:/n/fs/prl-chongyiz/exp_logs/ogbench_logs/sarsa_ifql_vib_gpi_offline2offline/20250519_sarsa_ifql_vib_gpi_offline2offline_cube-single-play-singletask-task2-v0*' \
      /n/fs/rl-chongyiz/exp_logs/ogbench_logs/sarsa_ifql_vib_gpi_offline2offline/
