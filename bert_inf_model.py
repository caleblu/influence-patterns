import tensorflow as tf
from official.nlp.modeling import layers as lys
from official.nlp.bert.bert_models import get_transformer_encoder, models
from checkpoint_utils import load_tf1_weights_cls, load_tf1_weights_core, \
    load_tf1_weights_emb, load_tf2_weights_cls, load_tf2_weights_core, load_tf2_weights_emb


class BERTInfModel():

    def __init__(
        self,
        bert_config,
        seq_length,
        max_predictions_per_seq=1,
        use_stop_gradient=False,
        model_compression=False,
    ):
        self.bert_config = bert_config
        self.seq_length = seq_length
        self.max_predictions_per_seq = max_predictions_per_seq
        self.use_stop_gradient = use_stop_gradient
        self.model_compression = model_compression

    def embedding_model_lm(
        self, initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)):
        embedding_layer = lys.OnDeviceEmbedding(
            vocab_size=self.bert_config.vocab_size,
            embedding_width=self.bert_config.hidden_size,
            initializer=initializer,
            name='word_embeddings')
        input_word_ids = tf.keras.layers.Input(shape=(self.seq_length,),
                                               name='input_word_ids',
                                               dtype=tf.int32)
        word_embeddings = embedding_layer(input_word_ids)
        embedding_model = tf.keras.Model(inputs=input_word_ids,
                                         outputs=word_embeddings)

        return embedding_model

    def build_model(
        self,
        lm_output_type='predictions',
        load_ckpt_mode='tf1_mlm',
        checkpoint='/longterm/USERNAME/data/bert/bert_models/uncased_L-12_H-768_A-12/bert_model.ckpt'
    ):
        embedding_model = self.embedding_model_lm()

        # print(embedding_model.layers[1].weights)
        transformer_encoder = get_transformer_encoder(
            self.bert_config,
            self.seq_length,
            # subj_idx=self.subj_idx,
            model_compression=self.model_compression,
            use_stop_gradient=self.use_stop_gradient)
        initializer = tf.keras.initializers.TruncatedNormal(
            stddev=self.bert_config.initializer_range)
        if load_ckpt_mode == 'tf1_mlm':

            decoder_model = models.BertPretrainer(
                num_classes=
                2,  # The next sentence prediction label has two classes.
                num_token_predictions=self.max_predictions_per_seq,
                sequence_length=self.seq_length,
                hidden_size=self.bert_config.hidden_size,
                initializer=initializer,
                embedding_table=embedding_model.layers[1].weights[0],
                output=lm_output_type)
            load_tf1_weights_emb(checkpoint, embedding_model)
            load_tf1_weights_core(checkpoint,
                                  transformer_encoder,
                                  num_layer=self.bert_config.num_hidden_layers)

            load_tf1_weights_cls(checkpoint, decoder_model)
        elif load_ckpt_mode == 'tf2_cls':
            decoder_model = models.BertClassifier(
                num_classes=2,
                sequence_length=self.seq_length,
                hidden_size=self.bert_config.hidden_size,
                initializer=initializer,
                output=lm_output_type)
            load_tf2_weights_emb(checkpoint, embedding_model)
            load_tf2_weights_core(checkpoint,
                                  transformer_encoder,
                                  num_layer=self.bert_config.num_hidden_layers)

            load_tf2_weights_cls(checkpoint, decoder_model)
        return embedding_model, transformer_encoder, decoder_model