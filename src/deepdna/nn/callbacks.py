import tensorflow as tf

class LearningRateStepScheduler(tf.keras.callbacks.Callback):
    def __init__(self, init_lr, max_lr, warmup_steps, end_steps):
        super().__init__()
        self.model: tf.keras.Model
        self.step = 0
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.end_steps = end_steps

    def on_train_batch_begin(self, batch, logs=None):
        self.step += 1
        if self.step < self.warmup_steps:
            lr = self.init_lr + (self.max_lr - self.init_lr)*(self.step/self.warmup_steps)
        else:
            lr = self.max_lr - (self.max_lr)*(
                (self.step - self.warmup_steps)/(self.end_steps - self.warmup_steps))
        self.model.optimizer.learning_rate.assign(lr)

    def on_epoch_end(self, epoch, logs=None):
        try:
            import wandb
        except ImportError:
            return
        if not wandb.run or wandb.run.disabled:
            return
        wandb.run.log({
            "epoch": epoch,
            "learning_rate": self.model.optimizer.learning_rate.numpy()
        })
