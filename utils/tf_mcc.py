
import tensorflow as tf
from tensorflow.keras import backend as K

@tf.keras.utils.register_keras_serializable(package = "custom_metrics")
class MCC(tf.keras.metrics.Metric):

    def __init__(self, name: str = "mcc", **kwargs,):

        #super().__init__(name = name, dtype = tf.float32)
        super(MCC, self).__init__(name=name, **kwargs)
        self.true_pos = self.add_weight(name = "tp", initializer = "zeros", dtype = tf.int64)
        self.true_neg = self.add_weight(name = "tn", initializer = "zeros", dtype = tf.int64)
        self.false_pos = self.add_weight(name = "fp", initializer = "zeros", dtype = tf.int64)
        self.false_neg = self.add_weight(name = "fn", initializer = "zeros", dtype = tf.int64)

    def update_state(self, y_true, y_pred, sample_weight = None):

        predicted = tf.cast(tf.greater(y_pred, 0.5), tf.int64)

        self.true_pos.assign_add(tf.math.count_nonzero(predicted * y_true))
        self.true_neg.assign_add(tf.math.count_nonzero((predicted - 1) * (y_true - 1)))
        self.false_pos.assign_add(tf.math.count_nonzero(predicted * (y_true - 1)))
        self.false_neg.assign_add(tf.math.count_nonzero((predicted - 1) * y_true))

    def result(self):

        x = tf.cast((self.true_pos + self.false_pos) * (self.true_pos + self.false_neg) * (self.true_neg + self.false_pos) * (self.true_neg + self.false_neg), tf.float32)
        mcc = tf.cast((self.true_pos * self.true_neg) - (self.false_pos * self.false_neg), tf.float32) / tf.sqrt(x)

        if (tf.math.is_nan(mcc)):

            mcc = tf.cast(0, tf.float32)

        return mcc

    def reset_state(self):

        for v in self.variables:

            K.set_value(v, tf.cast(0, tf.int64))

    def reset_states(self):
        # Required in Tensorflow < 2.5.0

        return self.reset_state()

