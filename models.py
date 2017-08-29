import tensorflow as tf

import constants as c
from net import nets, inception_resnet_v2, inception_resnet_v2_arg_scope
from net import get_exclude_names, get_logits_names

# import tensorflow.contrib.framework.python.ops
slim = tf.contrib.slim


class ClassifierModel(object):
    def __init__(self, session, data_labels_tensors,
                 multi_scales=c.MULTI_SCALES,
                 architecture='resnet_v2_50',
                 attention_names=''):

        """
        Initializes a ClassifierModel
        :param session: The Tensorflow Session.
        :param data_labels_tensors: The input tensor of model.
        :param multi_scales: images are resized such that the shorter side is in multi_scales when testing.

        :type session: tf.Session
        :type data_labels_tensors: tuple
        :type multi_scales: list
        """

        self.sess = session
        self.multi_scales = multi_scales

        assert architecture in nets.keys(), \
            'architecture {} must in {}'.format(architecture, nets.keys())

        self.architecture = architecture
        self.net = nets[architecture]
        self.attention_names = attention_names

        self.data = data_labels_tensors[0]
        self.labels = data_labels_tensors[1]
        self.attention_input = data_labels_tensors[2]
        self.attention_map = data_labels_tensors[3]
        self.logits = None
        self.end_points = None
        self.attention_ends = None
        self.attention_end_points = []

        self.saver = None

        self.loss = None
        self.global_loss = None
        self.attention_loss = None
        self.global_step = None

    def define_graph(self, is_training):
        ##
        # Network operators
        ##
        arg_scope = inception_resnet_v2_arg_scope
        model = inception_resnet_v2
        with slim.arg_scope(arg_scope()):

            # Classification
            _logits, _end_points = inception_resnet_v2(self.data, num_classes=1,
                                                       is_training=is_training, reuse=None,
                                                       scope=self.architecture,
                                                       attention_names=self.attention_names)
            self.logits = tf.squeeze(_logits, axis=[1], name='squeeze_logits')
            self.end_points = _end_points

            # Attention regression
            _logits, _end_points = inception_resnet_v2(self.attention_input, num_classes=1,
                                                       is_training=is_training, reuse=True,
                                                       scope=self.architecture,
                                                       attention_names=self.attention_names)
            #print(_end_points)

            self.attention_end_points = _end_points
        for variable in slim.get_variables(self.architecture):
            if ('Attention' in variable.name):
                print(variable)

        #for variable in slim.get_variables(self.architecture):
        #    print(variable)

        ##
        # Training
        ##
        with tf.name_scope('train'):
            # classification loss
            self.loss = tf.losses.mean_squared_error(
                labels=self.labels,
                predictions=self.logits,
            )

            # attention regression loss
            attention_losses = []
            for attention_name in self.attention_names:
                attention = self.attention_end_points['Attention-' + attention_name]
                size = attention.get_shape()[1:3]
                resize_map = tf.image.resize_images(self.attention_map, size=size)

                loss = tf.losses.mean_squared_error(
                    labels=resize_map,
                    predictions=attention
                )
                attention_losses.append(loss)
                tf.summary.image('Attention-' + attention_name, attention)
                tf.summary.scalar('Attention-MAX-' + attention_name, tf.reduce_max(attention))
                tf.summary.scalar('Attention-MIN-' + attention_name, tf.reduce_min(attention))

            tf.summary.image('LesionMap', self.attention_map)

            self.attention_loss = c.LAMBDA * tf.add_n(attention_losses, name='attention_loss')

            regularization_losses = tf.losses.get_regularization_losses()   # loss + regularization

            self.global_loss = tf.add_n([self.loss] + regularization_losses + [self.attention_loss], name='global_loss')
            self.global_step = tf.Variable(0, trainable=False, name='step')

            fine_tune_lr = c.FINE_TUNE_LR
            train_lr = c.LEARNING_RATE

            # only trainable variables
            m_net_train_vars = slim.get_trainable_variables(scope=self.architecture)
            # variables in attention
            attention_vars = [var for var in m_net_train_vars if 'Attention' in var.name]

            optimizer1 = tf.train.MomentumOptimizer(learning_rate=fine_tune_lr,
                                                    momentum=c.MOMENTUM, name='Optimizer1')
            optimizer2 = tf.train.MomentumOptimizer(learning_rate=train_lr,
                                                    momentum=c.MOMENTUM, name='Optimizer2')

            # variables in logits
            logits_vars = []
            for name in get_logits_names(self.architecture):
                logits_vars += slim.get_trainable_variables(name)

            # trainable variables except logits and attention
            main_variables = [var for var in m_net_train_vars
                              if var not in logits_vars and var not in attention_vars]

            assert len(m_net_train_vars) == len(main_variables) + len(logits_vars) + len(attention_vars), \
                'all trainable {} ! = main {} + logits {} + attention {}'.format(
                    len(m_net_train_vars), len(main_variables), len(logits_vars), len(attention_vars))

            # BN-means and variances
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                grads = tf.gradients(self.global_loss, main_variables + logits_vars + attention_vars)
                grads1 = grads[: len(main_variables) + len(logits_vars)]
                grads2 = grads[len(main_variables) + len(logits_vars):]

                train_op1 = optimizer1.apply_gradients(zip(grads1, main_variables + logits_vars))
                train_op2 = optimizer2.apply_gradients(zip(grads2, attention_vars),
                                                       global_step=self.global_step)
                self.train_op = tf.group(train_op1, train_op2, name='train_op')

            # add summaries
            tf.summary.scalar('learning_rate', train_lr)
            tf.summary.scalar('finetune_lr', fine_tune_lr)
            tf.summary.scalar('cls_loss', self.loss)
            tf.summary.scalar('attention_loss', self.attention_loss)
            tf.summary.scalar('global_loss', self.global_loss)

        # merge all summaries and write to file
        self.train_summaries_merge = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(c.SUMMARY_SAVE_DIR, graph=self.sess.graph)

    def use_pretrain(self, restore_path=None):
        if restore_path == 'imagenet':
            restore_path = self.net['ckpt']
            exclude_names = get_exclude_names(self.architecture)
        else:
            all_variables = slim.get_variables(self.architecture)
            exclude_names = ['train/step']
            for var in all_variables:
                if ('Attention' in var.name) or ('Optimizer' in var.name):
                    exclude_names.append(var.name)

        print('\tRestore from {}'.format(restore_path))

        variables_to_restore = slim.get_variables_to_restore(exclude=exclude_names)
        self.saver = tf.train.Saver(variables_to_restore)

        # for variable in variables_to_restore:
        #     print(variable)

        self.saver.restore(self.sess, restore_path)

    def train_step(self):
        """
        Runs a training step, compute loss and gradient, then update the parameters.

        :param batch_data: An array of shape [batch_size, c.CROP_HEIGHT, c.CROP_WIDTH, 3]
        :param batch_labels: An array of shape [bacth_size, c.NUM_CLASSES]
        :return: The global step.

        :type batch_data: numpy.ndarray
        :type batch_labels: list
        """

        _, loss, att_loss, global_loss, global_step, summaries = \
            self.sess.run([self.train_op, self.loss, self.attention_loss, self.global_loss,
                           self.global_step, self.train_summaries_merge])

        ##
        # User output
        ##
        if global_step % c.STATS_FREQ == 0:
            print('Iteration = {}, '
                  'cls loss = {:.6f}, '
                  'attention loss = {:.6f}, '
                  'global loss = {:.6f}'.format(global_step, loss, att_loss, global_loss))
        if global_step % c.SUMMARY_FREQ == 0:
            self.summary_writer.add_summary(summaries, global_step)
            print('Summaries saved at step = {}.'.format(global_step))

        return global_step

    def inference(self):
        """
        Inferences a mini-batch testing data and predicts the labels.
        :return: preds: An array of shape [?, 1]
        """
        logits = self.sess.run(self.logits)

        return logits

    def debug(self, batch_data, batch_labels):
        print('debug')
        feed_dict = {
            self.data: batch_data
        }

        logits = self.sess.run(self.logits, feed_dict=feed_dict)

        print(logits)
        print(batch_labels)

# fine_tune_lr = tf.train.exponential_decay(
#     learning_rate=c.FINE_TUNE_LR,
#     global_step=self.global_step,
#     decay_steps=c.DECAY_STEP * c.DECAY_EPOCH,  # decay after each 2 epochs
#     decay_rate=0.5,
#     staircase=True,
#     name='fine_tune_lr'
# )
#
# train_lr = tf.train.exponential_decay(
#     learning_rate=c.LEARNING_RATE,
#     global_step=self.global_step,
#     decay_steps=c.DECAY_STEP * c.DECAY_EPOCH,  # decay after each 2 epochs
#     decay_rate=0.5,
#     staircase=True,
#     name='train_lr'
# )
