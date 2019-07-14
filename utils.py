# -*- coding: utf-8 -*-




class LambdaLR:
    def __init__(self, max_epoch, decay_start_epoch):
        assert max_epoch > decay_start_epoch, "decay must start before the training session end!"
        self.max_epoch = max_epoch
        self.decay_start_epoch = decay_start_epoch
        
    def step(self, epoch):
        return 1 - max(0, epoch - self.decay_start_epoch) / (self.max_epoch - self.decay_start_epoch)

