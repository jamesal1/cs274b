vocab_size = 1000
zero_value = 0

reg_B = 10e-1 # usually 1e-3
reg_UV = 10e-1 # 1e-1 # usually 1e-3

reg_B = 0
reg_UV = 0

sampling_scheme = "complete" # "uniform" or "proportional" or "complete"
# sampling_scheme = "uniform"
proportion_positive = 0.3 # ignored if sampling scheme is uniform


logistic = False

co_is_identity = False