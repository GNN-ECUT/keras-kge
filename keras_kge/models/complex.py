import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Embedding, Flatten, Lambda, Multiply


class ComplEx(tensorflow.keras.Model):
    """
    Implementation of the `ComplEx` factorization model.
    """

    def __init__(self, nb_entities: int, nb_relationships: int, rank: int):
        """
        Constructor.

        :param nb_entities: total number of entities
        :param nb_relationships: total number of relationships
        :param rank: hyperparameter for the rank of factorization
        """
        super(ComplEx, self).__init__()

        self.nb_entities = nb_entities
        self.nb_relationships = nb_relationships
        self.rank = rank

        # Parameters

        self.entityEmbedding_real = Embedding(output_dim=self.rank, input_dim=self.nb_entities, input_length=1,
                                         name='embedding_entity_real')
        self.entityEmbedding_im = Embedding(output_dim=self.rank, input_dim=self.nb_entities, input_length=1,
                                       name='embedding_entity_im', )

        self.predicateEmbedding_real = Embedding(output_dim=self.rank, input_dim=self.nb_relationships, input_length=1,
                                            name='embedding_predicate_real')
        self.predicateEmbedding_im = Embedding(output_dim=self.rank, input_dim=self.nb_relationships, input_length=1,
                                          name='embedding_predicate_im')

    def call(self, inputs: list):

        s_input = inputs[0]
        p_input = inputs[1]
        o_input = inputs[2]

        s_embedding_real = self.entityEmbedding_real(s_input)
        s_embedding_real = Flatten()(s_embedding_real)

        s_embedding_im = self.entityEmbedding_im(s_input)
        s_embedding_im = Flatten()(s_embedding_im)

        p_embedding_real = self.predicateEmbedding_real(p_input)
        p_embedding_real = Flatten()(p_embedding_real)

        p_embedding_im = self.predicateEmbedding_im(p_input)
        p_embedding_im = Flatten()(p_embedding_im)

        o_embedding_real = self.entityEmbedding_real(o_input)
        o_embedding_real = Flatten()(o_embedding_real)

        o_embedding_im = self.entityEmbedding_im(o_input)
        o_embedding_im = Flatten()(o_embedding_im)

        rep1 = Multiply()([s_embedding_real, p_embedding_real, o_embedding_real])
        rep2 = Multiply()([s_embedding_im, p_embedding_real, o_embedding_im])
        rep3 = Multiply()([s_embedding_real, p_embedding_im, o_embedding_im])
        rep4 = Multiply()([s_embedding_im, p_embedding_im, o_embedding_real])

        rep = Lambda(lambda x: x[0] + x[1] + x[2] - x[3], output_shape=(None, self.rank))([rep1, rep2, rep3, rep4])

        out = Lambda(lambda x: K.exp(K.sum(x, axis=1)[:, None]), output_shape=(1,))(rep)

        return out
