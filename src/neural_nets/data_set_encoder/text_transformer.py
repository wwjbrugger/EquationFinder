import tensorflow as tf
from src.neural_nets.data_set_encoder.measurement_encoder_dummy import MeasurementEncoderDummy


class  TextTransformer(MeasurementEncoderDummy):
    def __init__(self, *args, **kwargs):
        super(TextTransformer, self).__init__(*args, **kwargs)
        # get your arguments from kwargs 
        self.num_blocks_text_transformer = kwargs['num_blocks_text_transformer']
        self.model=self.build_model(args, kwargs)
        


    def prepare_data(self, data):
        norm_frame = self.normalize(
            data_frame=data['data_frame'],
            approach=self.kwargs['normalize_approach']
        )
        # tokenizer here
        tensor_one_row = tf.reshape(tf.convert_to_tensor(norm_frame, dtype=tf.float32), -1)
        #add batch dimension
        tensor_one_row = tf.expand_dims(tensor_one_row, axis=0)
        return tensor_one_row

    def call(self, x, *args, **kwargs):
        #todo add call
        norm = tf.linalg.norm(X_flat_2, ord='euclidean', name=None, keepdims=True, axis=-1)
        out = X_flat_2 / norm
        return output
    
    def build_model(self,  *args, **kwargs):
        model = 'Your architecture'
        return model
