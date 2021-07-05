import tensorflow as tf
from blocks.label_block import LabelBranch
from blocks.box_detect_block import BoxBranch

class Heads(tf.keras.Model):
    def __init__(self, name, config):
        super().__init__(name=name)

        if config['train_labels']:
            self.label_branch = LabelBranch('label_branch', config)
        else:
            self.label_branch = None

        if config['train_boundingboxes']:
            self.box_branch = BoxBranch('box_branch', config)
        else:
            self.box_branch = None

    def call(self, x, train_batch_norm=False):
        results = {}

        if self.label_branch:
            labels = self.label_branch(x, train_batch_norm=train_batch_norm)
            results['pixelwise_labels'] = labels

        if self.box_branch:
            bb_targets, cls_targets, obj_targets, embedding_targets = \
                self.box_branch(x, train_batch_norm=train_batch_norm)
            results['bb_targets_offset'] = bb_targets
            results['bb_targets_cls'] = cls_targets
            results['bb_targets_objectness'] = obj_targets
            results['bb_targets_embedding'] = embedding_targets

        return results