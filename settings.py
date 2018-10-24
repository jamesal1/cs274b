vocab_size = 7000
zero_value = 0

# reg_B = 10e-1 # usually 1e-3
# reg_UV = 10e-1 # 1e-1 # usually 1e-3

reg_B = 1e-3
reg_UV = 1e-3

# reg_B = 0
# reg_UV = 0
sample_size_B = 50000 # usually 100000


# sampling_scheme = "complete" # "uniform" or "proportional" or "complete"
# sampling_scheme = "uniform"
sampling_scheme = "proportional"
proportion_positive = 0.5 # ignored if sampling scheme is uniform # usually 0.3


logistic = False

co_is_identity = False