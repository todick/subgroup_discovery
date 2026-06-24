import flowtorch.bijectors as bij

class Config_RQ1:
    def __init__(self):
        self.lr_flow = 5e-2
        self.lr_classifier = 2e-2
        self.alpha = 0.5
        self.lambd = 0.5
        self.pop_train_epochs = 2500
        self.subgroup_train_epochs = 1500
        self.final_fit_epochs = 0
        self.temperature = 0.2
        self.bin_deviation = 0.2
        self.use_weights = True
        self.seed = 0
        self.early_stopping_patience = 1000
        self.ncutpoints = 20
        self.beam_width = 40
        self.sd_depth = 8
        self.rsd_min_support = 0.025
        def flow_gen():
            return bij.Compose([bij.Spline(count_bins=12), bij.Spline(count_bins=12)])
        self.flow_gen = flow_gen

class Config_RQ3:
    def __init__(self):
        self.lr_flow = 5e-2
        self.lr_classifier = 2e-2
        self.alpha = 0.3
        self.lambd = 5.0
        self.pop_train_epochs = 2500
        self.subgroup_train_epochs = 2500
        self.final_fit_epochs = 0
        self.temperature = 0.2
        self.bin_deviation = 0.2
        self.use_weights = True
        self.seed = 0
        self.early_stopping_patience = 1000
        self.ncutpoints = 20
        self.beam_width = 40
        self.sd_depth = 5
        self.rsd_min_support = 0.025
        def flow_gen():
            return bij.Compose([bij.Spline(count_bins=12), bij.Spline(count_bins=12)])
        self.flow_gen = flow_gen
        self.n_rules = 5