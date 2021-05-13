import tensorflow as tf
from blocks.resnet_blocks import ResnetBranch
from models.layers.singleDecoder import SingleDecoder
from models.constants import L2_REGULARIZER_WEIGHT


class BoxDecoder(tf.keras.Model):
    """ A block that solves 
    
    """
    def __init__(self, name, config):
        super().__init__(name=name)

        boxes_per_pos = config['boxes_per_pos']
        self.config = config

        self.box_decoder = SingleDecoder('box_decoder', 4 * boxes_per_pos, 128)
        self.cls_decoder = SingleDecoder('cls_decoder', config['num_bb_classes'] * boxes_per_pos, 128)
        self.obj_decoder = SingleDecoder('obj_decoder', 2 * boxes_per_pos, 128)
        self.embedding_decoder = SingleDecoder('embedding_decoder',
                                               config['box_embedding_len'] * boxes_per_pos, 512)

    def call(self, x, train_batch_norm=False):
        n = x.get_shape().as_list()[0]
        # This is an ugly hack for saved_model. Without it fails to determine the batch size
        if n is None:
            n = 1

        box = self.box_decoder(x, train_batch_norm=train_batch_norm)
        cls = self.cls_decoder(x, train_batch_norm=train_batch_norm)
        obj = self.obj_decoder(x, train_batch_norm=train_batch_norm)
        embedding = self.embedding_decoder(x, train_batch_norm=train_batch_norm)
        embedding = tf.reshape(embedding, [n, -1, self.config['box_embedding_len']])
        embedding = tf.math.l2_normalize(embedding, axis=-1)
        embedding = tf.reshape(embedding, [n, -1, self.config['box_embedding_len'] * self.config['boxes_per_pos']])

        return box, cls, obj, embedding

class BoxBranch(tf.keras.Model):
    def __init__(self, name, config):
        super().__init__(name=name)
        self.core_branch = ResnetBranch('box_branch')
        self.decoder = BoxDecoder('decoder', config)

    def call(self, x, train_batch_norm=False):
        x = self.core_branch(x, train_batch_norm=train_batch_norm)

        res = []
        res += [self.decoder(x, train_batch_norm=train_batch_norm)]

        box = tf.concat([entry[0] for entry in res], axis=1)
        cls = tf.concat([entry[1] for entry in res], axis=1)
        obj = tf.concat([entry[2] for entry in res], axis=1)
        embedding = tf.concat([entry[3] for entry in res], axis=1)

        return box, cls, obj, embedding