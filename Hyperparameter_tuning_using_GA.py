import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from transformers import BertTokenizer, TFBertModel
from urllib.request import urlretrieve

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam

# Define JointIntentAndSlotFilling model
class JointIntentAndSlotFillingModel(tf.keras.Model):

    def __init__(self, intent_num_labels=None, slot_num_labels=None,
                 model_name="/kaggle/input/bert/tensorflow2/default/1/bert", dropout_prob=0.1):
        super().__init__(name="joint_intent_slot")
        self.bert = TFBertModel.from_pretrained(model_name)
        self.dropout = Dropout(dropout_prob)
        self.intent_classifier = Dense(intent_num_labels,
                                       name="intent_classifier")
        self.slot_classifier = Dense(slot_num_labels,
                                     name="slot_classifier")

    def call(self, inputs, **kwargs):
        # Get the output from the BERT model, which returns an object.
        bert_output = self.bert(inputs, **kwargs)
        # Extract `last_hidden_state` and `pooler_output` from the output object.
        sequence_output = bert_output.last_hidden_state  # (batch_size, max_length, output_dim)
        pooled_output = bert_output.pooler_output        # (batch_size, hidden_size)
        # Apply dropout to the sequence output for slot classification.
        sequence_output = self.dropout(sequence_output, training=kwargs.get("training", False))
        slot_logits = self.slot_classifier(sequence_output)
        # Apply dropout to the pooled output for intent classification.
        pooled_output = self.dropout(pooled_output, training=kwargs.get("training", False))
        intent_logits = self.intent_classifier(pooled_output)
    
        return slot_logits, intent_logits

# Sample dataset (replace with actual data)
X_train = np.random.rand(100, 10)  # Example: 100 samples, 10 features
y_train = np.random.randint(2, size=100)  # Binary classification labels
X_val = np.random.rand(20, 10)  # Example validation data
y_val = np.random.randint(2, size=20)  # Example validation labels

# Define the model creation function with learning_rate and batch_size as parameters
def create_model(learning_rate=0.001, batch_size=32):
    joint_model = JointIntentAndSlotFillingModel(
    intent_num_labels=len(intent_map), slot_num_labels=len(slot_map))
    opt = Adam(learning_rate=learning_rate, epsilon=1e-08)
    losses = [SparseCategoricalCrossentropy(from_logits=True),SparseCategoricalCrossentropy(from_logits=True)]
    metrics = [SparseCategoricalAccuracy('accuracy'),SparseCategoricalAccuracy('accuracy')]
    joint_model.compile(optimizer=opt, loss=losses, metrics=metrics)
    return joint_model

# Define the fitness function to evaluate the model with a specific learning rate and batch size
def fitness_function(hyperparameters):
    learning_rate, batch_size = hyperparameters
    model = create_model(learning_rate, batch_size)
    model.fit(X_train, y_train, epochs=3, batch_size=batch_size, verbose=0)  # Train for a few epochs
    _, accuracy = model.evaluate(X_val, y_val, verbose=0)  # Evaluate on validation set
    return accuracy  # The higher, the better

# Define GA components
def select_parents(population, fitness_scores):
    # Tournament selection (simplified)
    parents = random.choices(population, weights=fitness_scores, k=2)
    return parents

def crossover(parent1, parent2):
    # Single-point crossover (simple)
    return [parent1[0], parent2[1]], [parent2[0], parent1[1]]

def mutate(child, mutation_rate=0.1):
    # Mutate the learning rate or batch size
    if random.random() < mutation_rate:
        child[0] = random.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])  # Mutate learning rate
    if random.random() < mutation_rate:
        child[1] = random.choice([16, 32, 64, 128])  # Mutate batch size
    return child

# Define the population
population_size = 10
population = [
    [random.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1]), random.choice([16, 32, 64, 128])]
    for _ in range(population_size)
]

# Main GA loop
generations = 5
for generation in range(generations):
    # Evaluate the fitness of the population
    fitness_scores = [fitness_function(ind) for ind in population]
    
    # Select parents and create the next generation
    next_population = []
    for _ in range(population_size // 2):
        parent1, parent2 = select_parents(population, fitness_scores)
        child1, child2 = crossover(parent1, parent2)
        next_population.append(mutate(child1))
        next_population.append(mutate(child2))

    population = next_population
    print(f"Generation {generation}: Best fitness = {max(fitness_scores)}")

# Best solution
best_hyperparameters = population[fitness_scores.index(max(fitness_scores))]
print(f"Best hyperparameters: Learning Rate = {best_hyperparameters[0]}, Batch Size = {best_hyperparameters[1]}")
