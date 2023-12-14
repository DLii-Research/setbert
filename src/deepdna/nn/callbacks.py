import signal
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


class SafelyStopTrainingCallback(tf.keras.callbacks.Callback):
    """
    Safely stop a model from training using Ctrl+C keyboard interrupts.
    The first Ctrl+C will initiate a stop at the end of the current epoch.
    The second Ctrl+C will stop training immediately after the current batch.
    Subsequent Ctrl+C will be returned to the current signal handler.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stopping = False
        self.prev_signal_handler = signal.SIG_DFL

    def _restore_previous_signal_handler(self):
        if signal.getsignal(signal.SIGINT) != self._signal_handler:
            return
        signal.signal(signal.SIGINT, self.prev_signal_handler)

    def _signal_handler(self, sig, frame):
        if not self.stopping:
            self.stopping = True
            print("\nStopping training at the end of this epoch...")
            return
        print("\nStopping now...")
        self.model.stop_training = True
        self._restore_previous_signal_handler()

    def on_epoch_end(self, epoch: int, logs=None):
        if self.stopping:
            self.model.stop_training = True

    def on_train_begin(self, logs=None):
        print("Press Ctrl+C to stop training at the end of the epoch. Press again to stop immediately.")
        self.stopping = False
        self.prev_signal_handler = signal.signal(signal.SIGINT, self._signal_handler)

    def on_train_end(self, logs=None):
        self._restore_previous_signal_handler()
