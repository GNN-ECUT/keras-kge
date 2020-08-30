import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Embedding, Flatten, Lambda

from keras_kge.layers.embedding3d import Embedding3D


class Rescal(tf.keras.Model):
    """
    Implementation of the `RESCAL` factorization model.
    """

    def __init__(self, nb_entities: int, nb_relationships: int, rank: int):
        """
        Constructor.

        :param nb_entities: total number of entities
        :param nb_relationships: total number of relationships
        :param rank: hyperparameter for the rank of the factorization
        """
        super(Rescal, self).__init__()

        self.nb_entities = nb_entities
        self.nb_relationships = nb_relationships
        self.rank = rank

        # Parameters

        self.entityEmbedding = Embedding(output_dim=self.rank, input_dim=self.nb_entities, input_length=1, name='embedding_entity')

        R_init = np.repeat(np.diag(np.ones(self.rank))[np.newaxis, :, :], self.nb_relationships, axis=0)
        self.predicateEmbedding = Embedding3D(input_dim=self.nb_relationships, output_dim_1=self.rank, output_dim_2=self.rank,
                                       name='embedding_predicate', weights=[R_init])

    def call(self, inputs):

        s_input = inputs[0]
        p_input = inputs[1]
        o_input = inputs[2]

        s_embedding = self.entityEmbedding(s_input)
        s_embedding = Flatten()(s_embedding)

        p_embedding = self.predicateEmbedding(p_input)
        p_embedding = p_embedding[:, 0, :, :]

        o_embedding = self.entityEmbedding(o_input)
        o_embedding = Flatten()(o_embedding)

        def rescal_merge(x):
            sp = K.batch_dot(x[0], x[1], axes=1)
            spo = K.batch_dot(sp, x[2], axes=1)
            return K.sigmoid(spo)

        out = Lambda(function=lambda x: rescal_merge(x), output_shape=(1,))([s_embedding, p_embedding, o_embedding])

        return out
