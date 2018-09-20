import paddle.fluid as fluid


def fcn_net(x):
    conv1 = fluid.layers.conv2d(input=x, num_filters=36, filter_size=9, stride=1, padding=4,
                                param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")
    max_pool1 = fluid.layers.pool2d(input=conv1, pool_size=2, pool_type='max', pool_stride=2, global_pooling=False)
    conv2 = fluid.layers.conv2d(input=max_pool1, num_filters=72, filter_size=7, stride=1, padding=3,
                                param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")
    max_pool2 = fluid.layers.pool2d(input=conv2, pool_size=2, pool_type='max', pool_stride=2, global_pooling=False)
    conv3 = fluid.layers.conv2d(input=max_pool2, num_filters=36, filter_size=7, stride=1, padding=3,
                                param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")
    conv4 = fluid.layers.conv2d(input=conv3, num_filters=24, filter_size=7, stride=1, padding=3,
                                param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")
    conv5 = fluid.layers.conv2d(input=conv4, num_filters=16, filter_size=7, stride=1, padding=3,
                                param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")
    conv6 = fluid.layers.conv2d(input=conv5, num_filters=1, filter_size=1, stride=1,
                                param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")
    return conv6


def res_net_block1(x):
    conv1_1 = fluid.layers.conv2d(input=x, filter_size=3, num_filters=64, stride=1, padding=1,
                                  bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                                            regularizer=fluid.regularizer.L2DecayRegularizer(0.0),
                                                            learning_rate=2, trainable=True),
                                  param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")
    conv1_2 = fluid.layers.conv2d(input=conv1_1, filter_size=3, num_filters=64, stride=1, padding=1,
                                  bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                                            regularizer=fluid.regularizer.L2DecayRegularizer(0.0),
                                                            learning_rate=2, trainable=True),

                                  param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")

    max_pool1 = fluid.layers.pool2d(input=conv1_2, pool_size=2, pool_type='max', pool_stride=2, global_pooling=False)

    conv2_1 = fluid.layers.conv2d(input=max_pool1, filter_size=3, num_filters=128, stride=1, padding=1,
                                  bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                                            regularizer=fluid.regularizer.L2DecayRegularizer(0.0),
                                                            learning_rate=2, trainable=True),
                                  param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")
    conv2_2 = fluid.layers.conv2d(input=conv2_1, filter_size=3, num_filters=128, stride=1, padding=1,
                                  bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                                            regularizer=fluid.regularizer.L2DecayRegularizer(0.0),
                                                            learning_rate=2, trainable=True),
                                  param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")

    max_pool2 = fluid.layers.pool2d(input=conv2_2, pool_size=2, pool_type='max', pool_stride=2, global_pooling=False)

    conv3_1 = fluid.layers.conv2d(input=max_pool2, filter_size=3, num_filters=256, stride=1, padding=1,
                                  bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                                            regularizer=fluid.regularizer.L2DecayRegularizer(0.0),
                                                            learning_rate=2, trainable=True),
                                  param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")
    conv3_2 = fluid.layers.conv2d(input=conv3_1, filter_size=3, num_filters=256, stride=1, padding=1,
                                  bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                                            regularizer=fluid.regularizer.L2DecayRegularizer(0.0),
                                                            learning_rate=2, trainable=True),
                                  param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")
    conv3_3 = fluid.layers.conv2d(input=conv3_2, filter_size=3, num_filters=256, stride=1, padding=1,
                                  bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                                            regularizer=fluid.regularizer.L2DecayRegularizer(0.0),
                                                            learning_rate=2, trainable=True),
                                  param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")

    max_pool3 = fluid.layers.pool2d(input=conv3_3, pool_size=2, pool_type='max', pool_stride=2, global_pooling=False)

    conv4_1 = fluid.layers.conv2d(input=max_pool3, filter_size=3, num_filters=512, stride=1, padding=1,
                                  bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                                            regularizer=fluid.regularizer.L2DecayRegularizer(0.0),
                                                            learning_rate=2, trainable=True),
                                  param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")
    conv4_2 = fluid.layers.conv2d(input=conv4_1, filter_size=3, num_filters=512, stride=1, padding=1,
                                  bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                                            regularizer=fluid.regularizer.L2DecayRegularizer(0.0),
                                                            learning_rate=2, trainable=True),
                                  param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")
    conv4_3 = fluid.layers.conv2d(input=conv4_2, filter_size=3, num_filters=512, stride=1, padding=1,
                                  bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                                            regularizer=fluid.regularizer.L2DecayRegularizer(0.0),
                                                            learning_rate=2, trainable=True),
                                  param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")
    return conv4_3


def res_net_block2(x):
    max_pool4 = fluid.layers.pool2d(input=x, pool_size=2, pool_type='max', pool_stride=2, global_pooling=False)
    conv5_1 = fluid.layers.conv2d(input=max_pool4, filter_size=3, num_filters=512, stride=1, padding=1,
                                  bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                                            regularizer=fluid.regularizer.L2DecayRegularizer(0.0),
                                                            learning_rate=2, trainable=True),
                                  param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")
    conv5_2 = fluid.layers.conv2d(input=conv5_1, filter_size=3, num_filters=512, stride=1, padding=1,
                                  bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                                            regularizer=fluid.regularizer.L2DecayRegularizer(0.0),
                                                            learning_rate=2, trainable=True),
                                  param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")
    conv5_3 = fluid.layers.conv2d(input=conv5_2, filter_size=3, num_filters=512, stride=1, padding=1,
                                  bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                                            regularizer=fluid.regularizer.L2DecayRegularizer(0.0),
                                                            learning_rate=2, trainable=True),
                                  param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")
    max_pool5 = fluid.layers.pool2d(input=conv5_3, pool_size=3, pool_type='max', pool_stride=1, pool_padding=1,
                                    global_pooling=False)
    conv6_1 = fluid.layers.conv2d(input=max_pool5, filter_size=3, num_filters=512, stride=1, padding=1,
                                  bias_attr=fluid.ParamAttr(learning_rate=2, trainable=True,
                                                            regularizer=fluid.regularizer.L2DecayRegularizer(0.0)),
                                  param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")
    concat = fluid.layers.concat(input=[conv5_3, conv6_1], axis=1)
    deconv = fluid.layers.conv2d_transpose(input=concat, filter_size=2, num_filters=512, stride=2, padding=0,
                                           groups=512, bias_attr=False,
                                           param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(value=1.0)
                                                                      , trainable=False))
    concat2 = fluid.layers.concat(input=[x, deconv], axis=1)
    return concat2


def res_net(x):
    conv4_3 = res_net_block1(x)
    concat = res_net_block2(conv4_3)
    p_conv1 = fluid.layers.conv2d(input=concat, filter_size=3, num_filters=512, stride=1, padding=1,
                                  bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
							    regularizer=fluid.regularizer.L2DecayRegularizer(0.0),
                                                            learning_rate=2, trainable=True),
                                  param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")
    p_conv2 = fluid.layers.conv2d(input=p_conv1, filter_size=3, num_filters=256, stride=1, padding=1,
                                  bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
							    regularizer=fluid.regularizer.L2DecayRegularizer(0.0),
                                                            learning_rate=2, trainable=True),
                                  param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01), act="relu")
    p_conv3 = fluid.layers.conv2d(input=p_conv2, filter_size=1, num_filters=1, stride=1,
                                  bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
							    regularizer=fluid.regularizer.L2DecayRegularizer(0.0),
                                                            learning_rate=20, trainable=True),
                                  param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(loc=0.0, scale=0.01),
							   
                                                             learning_rate=10, trainable=True))
    return p_conv3
