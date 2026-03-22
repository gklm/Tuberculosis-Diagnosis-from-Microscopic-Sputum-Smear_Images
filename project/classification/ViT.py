import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Conv2D
from tensorflow.keras.models import Model

IMAGE_SIZE = 64
NUM_CHANNELS = 3
PATCH_SIZE = 16
PROJECTION_DIM = 64
NUM_HEADS = 8
TRANSFORMER_LAYERS = 8
MLP_HEAD_UNITS = [2048, 1024]
NUM_CLASSES = 2
DROPOUT_RATE = 0.1

class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size, projection_dim):
        super().__init__()
        self.conv = Conv2D(filters=projection_dim,
                           kernel_size=patch_size,
                           strides=patch_size,
                           padding='valid')

    def call(self, images):
        patches = self.conv(images)
        batch_size = tf.shape(images)[0]
        num_patches = tf.shape(patches)[1] * tf.shape(patches)[2]
        patches = tf.reshape(patches, (batch_size, num_patches, -1))
        return patches

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, projection_dim, dropout_rate):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim,
            dropout=dropout_rate
        )
        self.ffn = tf.keras.Sequential([
            Dense(projection_dim * 2, activation="relu"),
            Dense(projection_dim),
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training=False):
        attn_output = self.att(x, x)
        attn_output = self.dropout(attn_output, training=training)
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output, training=training)
        return self.norm2(x + ffn_output)

def create_vit_model():
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    x = PatchEmbedding(PATCH_SIZE, PROJECTION_DIM)(inputs)

    for _ in range(TRANSFORMER_LAYERS):
        x = TransformerBlock(NUM_HEADS, PROJECTION_DIM, DROPOUT_RATE)(x)

    x = LayerNormalization(epsilon=1e-6)(x[:, 0])
    x = Dense(MLP_HEAD_UNITS[0], activation="relu")(x)
    x = Dense(MLP_HEAD_UNITS[1], activation="relu")(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs)
