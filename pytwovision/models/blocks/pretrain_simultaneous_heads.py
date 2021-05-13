import tensorflow as tf
from models.constants import L2_REGULARIZER_WEIGHT
from blocks.resnet_blocks import ResnetBranch

class PretrainHead(tf.keras.Model):
    def __init__(self, name, config):
        super().__init__(name=name)

        self.core_branch = ResnetBranch('pretrain_branch')
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.final_conv = tf.keras.layers.Dense(config['num_pretrain_classes'],
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='final_conv')

    def call(self, x, train_batch_norm=False):
        x = self.core_branch(x, train_batch_norm=train_batch_norm)
        x = self.avg_pool(x)
        x = self.final_conv(x)
        return x