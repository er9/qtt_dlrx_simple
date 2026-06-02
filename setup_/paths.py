"""Output path configuration: defines the directories used for saving simulation
results and figures, and sets the matplotlib backend."""
save_dir = 'tmp_s/'  # '/pool001/erikaye/tns_pde_v2/'
main_dir = 'tmp_f/'  # '/pool001/erikaye/tns_pde_v2/'

import matplotlib
matplotlib.use('TkAgg')
