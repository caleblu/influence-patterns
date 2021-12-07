from tensorflow.python.training import py_checkpoint_reader
import tensorflow as tf

##reload the model weights from checkpoints since the model structure is changed due to modified code in /official


def load_tf2_weights_emb(tf2_checkpoint, embedding_model):
    reader = tf.train.load_checkpoint(tf2_checkpoint)
    embedding_model.set_weights([
        reader.get_tensor(
            'model/bert/embeddings/weight/.ATTRIBUTES/VARIABLE_VALUE')
    ])
    return embedding_model


def load_tf1_weights_emb(tf1_checkpoint, embedding_model):
    reader = py_checkpoint_reader.NewCheckpointReader(tf1_checkpoint)
    embedding_model.set_weights(
        [reader.get_tensor('bert/embeddings/word_embeddings')])
    return embedding_model


def load_tf2_weights_cls(tf2_checkpoint, decoder_model):
    reader = tf.train.load_checkpoint(tf2_checkpoint)
    tf2_cls_name_lst = [
        'model/classifier/kernel/.ATTRIBUTES/VARIABLE_VALUE',
        'model/classifier/bias/.ATTRIBUTES/VARIABLE_VALUE'
    ]
    decoder_model.layers[-1].set_weights(
        [reader.get_tensor(n) for n in tf2_cls_name_lst])


def load_tf1_weights_cls(tf1_checkpoint, decoder_model):
    reader = py_checkpoint_reader.NewCheckpointReader(tf1_checkpoint)
    tf2_cls_name_dict = {}

    tf2_cls_name_dict[-1] = [
        'cls/predictions/transform/dense/kernel',
        'cls/predictions/transform/dense/bias',
        'cls/predictions/transform/LayerNorm/gamma',
        'cls/predictions/transform/LayerNorm/beta',
        'cls/predictions/output_bias'
    ]

    decoder_model.layers[-1].set_weights(
        [reader.get_tensor(n) for n in tf2_cls_name_dict[-1]])


def load_tf2_weights_core(tf2_checkpoint, core_model, num_layer):
    reader = tf.train.load_checkpoint(tf2_checkpoint)
    layer_name_lst = [(i, e.name) for i, e in enumerate(core_model.layers)]
    tf2_core_name_dict = {}

    def locate_index(substring, exclude=None):
        for i, e in layer_name_lst:
            if exclude:
                flag = substring in e and exclude not in e
            else:
                flag = substring in e
            if flag:
                # print(substring, e)
                return i

    # tf2_core_name_dict[1] = ['bert/embeddings/word_embeddings']
    tf2_core_name_dict[locate_index('position_embedding')] = [
        'model/bert/embeddings/embeddings/.ATTRIBUTES/VARIABLE_VALUE'
    ]
    tf2_core_name_dict[locate_index('type_embeddings')] = [
        'model/bert/embeddings/token_type_embeddings/.ATTRIBUTES/VARIABLE_VALUE'
    ]
    tf2_core_name_dict[locate_index('embeddings/layer_norm')] = [
        'model/bert/embeddings/LayerNorm/gamma/.ATTRIBUTES/VARIABLE_VALUE',
        'model/bert/embeddings/LayerNorm/beta/.ATTRIBUTES/VARIABLE_VALUE'
    ]
    attn = [
        'model/bert/encoder/layer/{}/attention/self_attention/query/kernel/.ATTRIBUTES/VARIABLE_VALUE',
        'model/bert/encoder/layer/{}/attention/self_attention/query/bias/.ATTRIBUTES/VARIABLE_VALUE',
        'model/bert/encoder/layer/{}/attention/self_attention/key/kernel/.ATTRIBUTES/VARIABLE_VALUE',
        'model/bert/encoder/layer/{}/attention/self_attention/key/bias/.ATTRIBUTES/VARIABLE_VALUE',
        'model/bert/encoder/layer/{}/attention/self_attention/value/kernel/.ATTRIBUTES/VARIABLE_VALUE',
        'model/bert/encoder/layer/{}/attention/self_attention/value/bias/.ATTRIBUTES/VARIABLE_VALUE'
    ]
    attn_output = [
        'model/bert/encoder/layer/{}/attention/dense_output/dense/kernel/.ATTRIBUTES/VARIABLE_VALUE',
        'model/bert/encoder/layer/{}/attention/dense_output/dense/bias/.ATTRIBUTES/VARIABLE_VALUE'
    ]
    attn_output_norm = [
        'model/bert/encoder/layer/{}/attention/dense_output/LayerNorm/gamma/.ATTRIBUTES/VARIABLE_VALUE',
        'model/bert/encoder/layer/{}/attention/dense_output/LayerNorm/beta/.ATTRIBUTES/VARIABLE_VALUE'
    ]
    inte = [
        'model/bert/encoder/layer/{}/intermediate/dense/kernel/.ATTRIBUTES/VARIABLE_VALUE',
        'model/bert/encoder/layer/{}/intermediate/dense/bias/.ATTRIBUTES/VARIABLE_VALUE'
    ]
    output = [
        'model/bert/encoder/layer/{}/bert_output/dense/kernel/.ATTRIBUTES/VARIABLE_VALUE',
        'model/bert/encoder/layer/{}/bert_output/dense/bias/.ATTRIBUTES/VARIABLE_VALUE'
    ]
    output_norm = [
        'model/bert/encoder/layer/{}/bert_output/LayerNorm/gamma/.ATTRIBUTES/VARIABLE_VALUE',
        'model/bert/encoder/layer/{}/bert_output/LayerNorm/beta/.ATTRIBUTES/VARIABLE_VALUE'
    ]

    tf2_core_name_dict[locate_index('pooler_transform')] = [
        'model/bert/pooler/dense/kernel/.ATTRIBUTES/VARIABLE_VALUE',
        'model/bert/pooler/dense/bias/.ATTRIBUTES/VARIABLE_VALUE'
    ]
    for l in range(num_layer):
        tf2_core_name_dict[locate_index(
            'self_attention_{}'.format(l))] = [s.format(l) for s in attn]
        tf2_core_name_dict[locate_index(
            'self_attention_output_{}'.format(l))] = [
                s.format(l) for s in attn_output
            ]
        tf2_core_name_dict[locate_index(
            'self_attention_layer_norm_{}'.format(l))] = [
                s.format(l) for s in attn_output_norm
            ]
        tf2_core_name_dict[locate_index(
            'intermediate_{}'.format(l))] = [s.format(l) for s in inte]
        tf2_core_name_dict[locate_index('output_{}'.format(l), 'attention')] = [
            s.format(l) for s in output
        ]

        tf2_core_name_dict[locate_index('output_layer_norm_{}'.format(l))] = [
            s.format(l) for s in output_norm
        ]
    for a in tf2_core_name_dict:
        weight_shapes = [w.shape for w in core_model.layers[a].weights]
        core_model.layers[a].set_weights([
            reader.get_tensor(n).reshape(s)
            for (n, s) in zip(tf2_core_name_dict[a], weight_shapes)
        ])


def load_tf1_weights_core(tf1_checkpoint, core_model, num_layer):
    reader = py_checkpoint_reader.NewCheckpointReader(tf1_checkpoint)
    layer_name_lst = [(i, e.name) for i, e in enumerate(core_model.layers)]

    def locate_index(substring, exclude=None):
        for i, e in layer_name_lst:
            if exclude:
                flag = substring in e and exclude not in e
            else:
                flag = substring in e
            if flag:
                return i

    tf2_core_name_dict = {}
    tf2_core_name_dict[locate_index('position_embedding')] = [
        'bert/embeddings/position_embeddings'
    ]
    tf2_core_name_dict[locate_index('type_embeddings')] = [
        'bert/embeddings/token_type_embeddings'
    ]
    tf2_core_name_dict[locate_index('embeddings/layer_norm')] = [
        'bert/embeddings/LayerNorm/gamma', 'bert/embeddings/LayerNorm/beta'
    ]
    attn = [
        'bert/encoder/layer_{}/attention/self/query/kernel',
        'bert/encoder/layer_{}/attention/self/query/bias',
        'bert/encoder/layer_{}/attention/self/key/kernel',
        'bert/encoder/layer_{}/attention/self/key/bias',
        'bert/encoder/layer_{}/attention/self/value/kernel',
        'bert/encoder/layer_{}/attention/self/value/bias'
    ]
    attn_output = [
        'bert/encoder/layer_{}/attention/output/dense/kernel',
        'bert/encoder/layer_{}/attention/output/dense/bias'
    ]
    attn_output_norm = [
        'bert/encoder/layer_{}/attention/output/LayerNorm/gamma',
        'bert/encoder/layer_{}/attention/output/LayerNorm/beta'
    ]
    inte = [
        'bert/encoder/layer_{}/intermediate/dense/kernel',
        'bert/encoder/layer_{}/intermediate/dense/bias'
    ]
    output = [
        'bert/encoder/layer_{}/output/dense/kernel',
        'bert/encoder/layer_{}/output/dense/bias'
    ]
    output_norm = [
        'bert/encoder/layer_{}/output/LayerNorm/gamma',
        'bert/encoder/layer_{}/output/LayerNorm/beta'
    ]
    for l in range(num_layer):
        tf2_core_name_dict[locate_index(
            'self_attention_{}'.format(l))] = [s.format(l) for s in attn]
        tf2_core_name_dict[locate_index(
            'self_attention_output_{}'.format(l))] = [
                s.format(l) for s in attn_output
            ]
        tf2_core_name_dict[locate_index(
            'self_attention_layer_norm_{}'.format(l))] = [
                s.format(l) for s in attn_output_norm
            ]
        tf2_core_name_dict[locate_index(
            'intermediate_{}'.format(l))] = [s.format(l) for s in inte]
        tf2_core_name_dict[locate_index('output_{}'.format(l), 'attention')] = [
            s.format(l) for s in output
        ]

        tf2_core_name_dict[locate_index('output_layer_norm_{}'.format(l))] = [
            s.format(l) for s in output_norm
        ]
    for a in tf2_core_name_dict:
        weight_shapes = [w.shape for w in core_model.layers[a].weights]
        core_model.layers[a].set_weights([
            reader.get_tensor(n).reshape(s)
            for (n, s) in zip(tf2_core_name_dict[a], weight_shapes)
        ])