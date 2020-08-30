import tensorflow.keras
from tensorflow.keras.layers import Embedding, Flatten, Concatenate, Dense


class MultiMLP(tensorflow.keras.Model):
    """
    Implementation of the `Multiway Multi-Layer-Perceptron` model.
    """

    def __init__(self, nb_entities: int, nb_relationships: int, embedding_dim: int, hidden_units: int):
        """
        Constructor.

        :param nb_entities: total number of entities
        :param nb_relationships: total number of relationships
        :param embedding_dim: hyperparameter for the dimension of the embeddings
        :param hidden_units: hyperparameter for the hidden units of the MLP
        """
        super(MultiMLP, self).__init__()

        self.nb_entities = nb_entities
        self.nb_relationships = nb_relationships
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units

        # Parameters

        self.entityEmbedding = Embedding(output_dim=self.embedding_dim, input_dim=self.nb_entities, input_length=1, name='embedding_entity')
        self.predicateEmbedding = Embedding(output_dim=self.embedding_dim, input_dim=self.nb_relationships, input_length=1, name='embedding_predicate')
        self.hiddenLayer = Dense(units=self.hidden_units, input_dim=self.embedding_dim * 3, name='hidden_layer', activation='tanh')
        self.outputLayer = Dense(units=1, input_dim=self.hidden_units, name='output_layer', activation='sigmoid')


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

        rep = Concatenate()([s_embedding, p_embedding, o_embedding])

        h = self.hiddenLayer(rep)

        out = self.outputLayer(h)

        return out
