import numpy as np

from keras_kge.models import DistMult, ComplEx, MultiMLP, Rescal
from keras_kge.utils.sampling import sample_corrupted_triples

# Prepare data

kinship_data_pos = np.loadtxt('data/kinship.csv', comments='#', delimiter=',', dtype='int')

np.random.shuffle(kinship_data_pos)
kinship_data_neg = sample_corrupted_triples(kinship_data_pos, 2, 1)

nb_train_pos = round(0.8*kinship_data_pos.shape[0])
nb_test_pos = kinship_data_pos.shape[0] - nb_train_pos

nb_entities = kinship_data_pos[:, [0, 2]].max()
nb_relations = kinship_data_pos[:, [1]].max()

train_triples_pos, test_triples_pos = kinship_data_pos[:nb_train_pos, :], kinship_data_pos[nb_train_pos:, :]
train_triples_neg, test_triples_neg = kinship_data_neg[:nb_train_pos, :], kinship_data_neg[nb_train_pos:, :]

X_train = np.vstack([train_triples_pos, train_triples_neg])
y_train = np.array([1] * nb_train_pos + [0] * nb_train_pos)

X_test = np.vstack([test_triples_pos, test_triples_neg])
y_test = np.array([1] * nb_test_pos + [0] * nb_test_pos)

# Build models

rank = 100
multnn_hidden_units = 30

distmult = DistMult(nb_entities + 1, nb_relations + 1, rank)
complex = ComplEx(nb_entities + 1, nb_relations + 1, rank)
multnn = MultiMLP(nb_entities + 1, nb_relations + 1, rank, multnn_hidden_units)
rescal = Rescal(nb_entities + 1, nb_relations + 1, rank)

distmult.compile(optimizer='adam', loss='binary_crossentropy')
complex.compile(optimizer='adam', loss='binary_crossentropy')
multnn.compile(optimizer='adam', loss='binary_crossentropy')
rescal.compile(optimizer='adam', loss='binary_crossentropy')

# Train models

distmult.fit(x=[X_train[:, 0], X_train[:, 1], X_train[:, 2]], y=y_train, epochs=40)
complex.fit(x=[X_train[:, 0], X_train[:, 1], X_train[:, 2]], y=y_train, epochs=40)
multnn.fit(x=[X_train[:, 0], X_train[:, 1], X_train[:, 2]], y=y_train, epochs=80)
rescal.fit(x=[X_train[:, 0], X_train[:, 1], X_train[:, 2]], y=y_train, epochs=80)


# Evaluate models

y_distmult = distmult.predict(x=[X_test[:, 0], X_test[:, 1], X_test[:, 2]]).flatten()
y_complex = complex.predict(x=[X_test[:, 0], X_test[:, 1], X_test[:, 2]]).flatten()
y_multnn = multnn.predict(x=[X_test[:, 0], X_test[:, 1], X_test[:, 2]]).flatten()
y_rescal = rescal.predict(x=[X_test[:, 0], X_test[:, 1], X_test[:, 2]]).flatten()

accuracy_distmult = sum((y_distmult > 0.5) == y_test) / y_test.shape[0]
accuracy_complex = sum((y_complex > 0.5) == y_test) / y_test.shape[0]
accuracy_multnn = sum((y_multnn > 0.5) == y_test) / y_test.shape[0]
accuracy_rescal = sum((y_rescal > 0.5) == y_test) / y_test.shape[0]

print("Test accuracy DistMult: %.2f" % accuracy_distmult)
print("Test accuracy ComplEx: %.2f" % accuracy_complex)
print("Test accuracy MultNN: %.2f" % accuracy_multnn)
print("Test accuracy Rescal: %.2f" % accuracy_rescal)
