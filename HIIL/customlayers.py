import tensorflow as tf
import numpy

class FeatureScalingNormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return (x - tf.keras.backend.mean(x, 1, keepdims=True)) / tf.keras.backend.std(x, 1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape


class NanFeatureScalingNormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        valid_values = tf.keras.backend.equal(x, x)
        valid_values_float32 = tf.keras.backend.cast(valid_values, 'float32')

        x_no_nan = tf.keras.backend.switch(valid_values, x, tf.keras.backend.zeros_like(x))
        valid_count = tf.keras.backend.sum(valid_values_float32, 1, keepdims=True)

        means = tf.keras.backend.sum(x_no_nan*valid_values_float32, 1, keepdims=True) / valid_count
        means_rep = tf.keras.backend.repeat_elements(means, x_no_nan.shape[-1], -1)
        x_mean_subst = tf.keras.backend.switch(valid_values, x, means_rep)

        sqr = tf.keras.backend.pow((x_mean_subst - means), 2)
        std = tf.keras.backend.sqrt(tf.keras.backend.sum(sqr, 1, keepdims=True) / valid_count)

        return (x_mean_subst - means) / std


    def compute_output_shape(self, input_shape):
        return input_shape

class NanDropout(tf.keras.layers.Layer):
    def __init__(self, probability, **kwargs):
        self.probability = probability
        super().__init__(**kwargs)

    def get_config(self):
        config = {'probability': self.probability}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):
        nanmatrix = tf.keras.backend.ones_like(x) * numpy.nan
        #If not in lerning phase, will always be false...
        random_probability = tf.keras.backend.random_uniform(tf.keras.backend.shape(x))
        prob = (self.probability - 1 + tf.keras.backend.cast(tf.keras.backend.learning_phase(), 'float32'))
        return tf.keras.backend.switch(random_probability < prob, nanmatrix, x)


    def compute_output_shape(self, input_shape):
        return input_shape

class HIILLayer(tf.keras.layers.Layer):
    def __init__(self, input_wavelengths, output_wavelength_min, output_wavelength_max, output_wavelength_tree_depth, acceptable_missing=0.1, **kwargs):
        if type(input_wavelengths) == dict:
            assert input_wavelengths['type'] == 'ndarray'
            input_wavelengths = input_wavelengths['value']
        self.input_wavelengths = numpy.array(input_wavelengths)
        self.output_wavelength_min = output_wavelength_min
        self.output_wavelength_max = output_wavelength_max
        self.output_wavelength_tree_depth = output_wavelength_tree_depth
        self.acceptable_missing = acceptable_missing
        super().__init__(**kwargs)

    def get_config(self):
        config = {'input_wavelengths': self.input_wavelengths,
                  'output_wavelength_min': self.output_wavelength_min,
                  'output_wavelength_max': self.output_wavelength_max,
                  'output_wavelength_tree_depth': self.output_wavelength_tree_depth}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        middles = (self.input_wavelengths[1:] + self.input_wavelengths[:-1]) / 2
        b_1 = numpy.concatenate([[2 * self.input_wavelengths[0] - middles[0]], middles])
        b_2 = numpy.concatenate([middles, [2 * self.input_wavelengths[-1] - middles[-1]]])

        factor_matrix = []

        #Add an all-1 line
        f_1 = numpy.clip((b_2 - self.output_wavelength_min) / (b_2-b_1),0,1)
        f_2 = numpy.clip((self.output_wavelength_max - b_1) / (b_2-b_1),0,1)
        factor_matrix.append((f_1 + f_2 - 1) * (b_2-b_1))

        for tree_depth in range(self.output_wavelength_tree_depth):
            step_size = (self.output_wavelength_max - self.output_wavelength_min) / (2 ** tree_depth)
            for step_id in range(2**tree_depth):
                step_start = self.output_wavelength_min + step_id * step_size
                step_middle = self.output_wavelength_min + step_id * step_size + step_size / 2
                step_end = self.output_wavelength_min + step_id * step_size + step_size

                f_1 = numpy.clip((b_2 - step_start) / (b_2-b_1),0,1)
                f_2 = numpy.clip((step_middle - b_1)/(b_2-b_1),0,1)

                p1 = (f_1 + f_2 - 1) * (b_2-b_1)

                f_1 = numpy.clip((b_2 - step_middle) / (b_2-b_1),0,1)
                f_2 = numpy.clip((step_end - b_1)/(b_2-b_1),0,1)

                p2 = -(f_1 + f_2 - 1) * (b_2-b_1)
                factor_matrix.append((p1+p2))

        factor_matrix = numpy.array(factor_matrix).T

        self._factors_numpy = factor_matrix
        factor_matrix_neq_0 = factor_matrix != 0
        self._factors_one_numpy = factor_matrix_neq_0 / factor_matrix_neq_0.sum(0, keepdims=True)
        self._factors = tf.keras.backend.constant(self._factors_numpy)
        self._factors_one = tf.keras.backend.constant(self._factors_one_numpy)

        super().build(input_shape)

    def call(self, x):
        valid_values = tf.keras.backend.equal(x, x)
        valid_values_float32 = tf.keras.backend.cast(valid_values, 'float32')

        x_no_nan = tf.keras.backend.switch(valid_values, x, tf.keras.backend.zeros_like(x))

        output = tf.keras.backend.dot(x_no_nan, self._factors)

        output_invalid = tf.keras.backend.dot(valid_values_float32, self._factors_one)

        return tf.keras.backend.switch(output_invalid < (1-self.acceptable_missing), tf.keras.backend.zeros_like(output), output)

class LabelDenseLayer(tf.keras.layers.Dense):
    def __init__(self, labels, **kwargs):
        if type(labels) == dict:
            assert labels['type'] == 'list'
            labels = labels['value']
        self._labels = labels
        kw = kwargs.copy()
        kw['units'] = len(self.labels)
        super().__init__(**kw)

    @property
    def labels(self):
        return self._labels.copy()

    def get_config(self):
        config = super().get_config()
        del config['units']
        config['labels'] = self.labels
        return config
