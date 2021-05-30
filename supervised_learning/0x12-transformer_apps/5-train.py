#!/usr/bin/env python3
"""
Defines function that creates and trains a transformer for machine translation
of Portuguese to English using previously created dataset
"""


import tensorflow.compat.v2 as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class LearningSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Establishes learning rate schedule for training transformer model

    Utilizes given function for learning rate:
    l_rate = (dm ** -0.5) * min(
                     (step_num ** -0.5), (step_num * warmup_steps ** -1.5))
    """
    def __init__(self, dm, warmup_steps=4000):
        """
        Class constructor

        parameters:
            dm [int]:
                dimensionality of the model
            warmup_steps [int]:
                number of warmup steps, default=4000
        """
        super(LearningSchedule, self).__init__()

        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
        Evaluates the learning rate for the given step
        """
        rsqrt_dm = tf.math.rsqrt(self.dm)
        rsqrt_step_arg = tf.math.rsqrt(step)
        warmup_step_arg = step * (self.warmup_steps ** -1.5)
        l_rate = rsqrt_dm * tf.math.minimum(rsqrt_step_arg, warmup_step_arg)
        return l_rate


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Creates and trains a transformer for machine translation
        of Portuguese to English using previously created dataset

    parameters:
        N [int]:
            number of blocks in the encoder and decoder
        dm [int]:
            dimensionality of the model
        h [int]:
            number of heads
        hidden [int]:
            number of hidden units in the fully connected layers
        max_len [int]:
            maximum number of tokens per sequence
        batch_size [int]:
            batch size used for training
        epochs [int]:
            number of epochs to train for
    returns:
        the trained model
    """
    data = Dataset(batch_size, max_len)
    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2
    # encoder = data.tokenizer_pt
    # decoder = data.tokenizer_en

    transformer = Transformer(N, dm, h, hidden,
                              input_vocab, target_vocab,
                              max_len, max_len)

    learning_rate = LearningSchedule(dm)

    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
    losses = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                           reduction='none')

    def loss_function(actual, prediction):
        """
        Calculate the loss from actual value and the prediction
        """
        mask = tf.math.logical_not(tf.math.equal(actual, 0))
        loss = losses(actual, prediction)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def accuracy_function(actual, prediction):
        """
        Calculates the accuracy of the prediction
        """
        accuracies = tf.equal(actual, tf.argmax(prediction, axis=2))
        mask = tf.math.logical_not(tf.math.equal(actual, 0))
        accuracies = tf.math.logical_and(mask, accuracies)
        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    for epoch in range(epochs):
        batch = 0
        for (input, target) in data.data_train:
            target_input = target[:, :-1]
            target_actual = target[:, 1:]
            encoder_mask, look_ahead_mask, decoder_mask = create_masks(
                input, target_input)
            with tf.GradientTape() as tape:
                prediction = transformer(input, target_input, True,
                                         encoder_mask, look_ahead_mask,
                                         decoder_mask)
                loss = loss_function(target_actual, prediction)
            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(
                gradients, transformer.trainable_variables))
            t_loss = train_loss(loss)
            t_accuracy = train_accuracy(target_actual, prediction)
            if batch % 50 is 0:
                print("Epoch {}, batch {}: loss {} accuracy {}".format(
                    epoch, batch, t_loss, t_accuracy))
            batch += 1
        print("Epoch {}: loss {} accuracy {}".format(
            epoch, t_loss, t_accuracy))
    return transformer
