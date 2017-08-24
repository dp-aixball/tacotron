import tensorflow as tf


# Default hyperparameters:
hparams = tf.contrib.training.HParams(
  # Text:
  force_lowercase=False,
  expand_abbreviations=False,
  use_cmudict=False,

  # Audio:
  num_mels=80,
  num_freq=1025,
  #num_freq=1000,
  sample_rate=20000,
  #sample_rate=16000,
  frame_length_ms=50,
  frame_shift_ms=12.5,
  preemphasis=0.97,
  min_level_db=-100,
  ref_level_db=20,

  # Model:
  # TODO: add more configurable hparams
  outputs_per_step=5,

  # Training:
  batch_size=16, #32,
  adam_beta1=0.9,
  adam_beta2=0.999,
  initial_learning_rate=0.002,
  decay_learning_rate=True,

  # Eval:
  max_iters=int(200*2*1.5*1.5*1.5*1.5*1.5),
  griffin_lim_iters=60
)


def hparams_debug_string():
  values = hparams.values()
  hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
  return 'Hyperparameters:\n' + '\n'.join(hp)
