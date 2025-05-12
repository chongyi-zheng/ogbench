rsync -av --progress --exclude='*.pkl' \
      -e ssh \
      'cz8792@neuronic.cs.princeton.edu:/n/fs/prl-chongyiz/exp_logs/ogbench_logs/sarsa_ifql_vib_gpi_offline2offline/20250511_sarsa_ifql_vib_gpi_offline2offline_quadruped*' \
      /n/fs/rl-chongyiz/exp_logs/ogbench_logs/sarsa_ifql_vib_gpi_offline2offline/

