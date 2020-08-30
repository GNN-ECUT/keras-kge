import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Multiply, Embedding, Flatten, Lambda


class DistMult(tf.keras.Model):
    """
    Implementation of the `DistMult` factorization model.
    """

    def __init__(self, nb_entities: int, nb_relationships: int, rank: int):
        """
        Constructor.

        :param nb_entities: total number of entities
        :param nb_relationships: total number of relationships
        :param rank: hyperparameter for the rank of the factorization
        """
        super(DistMult, self).__init__()

        self.nb_entities = nb_entities
        self.nb_relationships = nb_relationships
        self.rank = rank

        # Parameters

        self.entityEmbedding = Embedding(output_dim=self.rank,
                                    input_dim=self.nb_entities,
                                    input_length=1,
                                    name='embedding_entity')

        self.predicateEmbedding = Embedding(output_dim=self.rank,
                                       input_dim=self.nb_relationships,
                                       input_length=1,
                                       name='embedding_relationship')

    def call(self, inputs):

        s_input = inputs[0]
        p_input = inputs[1]
        o_input = inputs[2]

        s_embedding = self.entityEmbedding(s_input)
        s_embedding = Flatten()(s_embedding)

        p_embedding = self.predicateEmbedding(p_input)
        p_embedding = Flatten()(p_embedding)

        o_embedding = self.entityEmbedding(o_input)
        o_embedding = Flatten()(o_embedding)

        rep = Multiply()([s_embedding, p_embedding, o_embedding])

        out = Lambda(lambda x: K.sigmoid(K.sum(x, axis=1)[:, None]), output_shape=(1,))(rep)

        return out
