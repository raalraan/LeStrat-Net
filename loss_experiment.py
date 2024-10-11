import tensorflow as tf


def loss_setup(loss='categorical_crossentropy', mod=None, alpha=1.0, cep=None):
    if cep is None:
        if loss == 'binary_crossentropy':
            cep = 7
        else:
            cep = 20

    clip_eps = 10**-cep

    if loss == 'categorical_crossentropy':

        @tf.function()
        def losstyp(y_true, y_pred):
            y_pred_cl = tf.clip_by_value(y_pred, clip_eps, 1)
            res = tf.math.log(y_pred_cl[y_true == 1])
            return -res

    elif loss == 'binary_crossentropy':

        @tf.function()
        def losstyp(y_true, y_pred):
            y_pred_cl = tf.clip_by_value(y_pred, clip_eps, 1 - clip_eps)
            bce1 = tf.math.log(y_pred_cl)*y_true
            bce2 = tf.math.log(1.0 - y_pred_cl)*(1.0 - y_true)
            return -tf.math.reduce_sum(bce1 + bce2, axis=1)

    elif loss == 'squared_hinge':

        @tf.function()
        def losstyp(y_true, y_pred):
            return tf.math.reduce_sum((1 - y_true*y_pred)**2, axis=1)

    else:
        raise ValueError(
            "Option `loss` must be one of:\n"
            "    categorical_crossentropy\n"
            "    binary_crossentropy\n"
            "    squared_hinge\n"
        )

    if mod == 0 or mod is None:
        @tf.function()
        def getmod(y_true, y_pred):
            return 1.0
    else:
        @tf.function()
        def getrdiff(y_true, y_pred):
            y_true_adj = (y_true + 1)/2
            y_pred_adj = (y_pred + 1)/2

            y_true_r = tf.math.round(y_true_adj)
            y_pred_r = tf.math.round(y_pred_adj)

            r_true = tf.math.reduce_sum(y_true_r, axis=1)
            r_pred = tf.math.reduce_sum(y_pred_r, axis=1)

            r_diff = tf.math.abs(r_true - r_pred)

            return r_diff**alpha

        if mod in [1, 3, 5]:
            @tf.function()
            def getmod(y_true, y_pred):
                return getrdiff(y_true, y_pred)
        elif mod in [2, 4, 6]:
            @tf.function()
            def getmod(y_true, y_pred):
                return 1 + getrdiff(y_true, y_pred)

    if mod in [0, 1, 2] or mod is None:
        @tf.function()
        def myloss(y_true, y_pred):
            loss1 = losstyp(y_true, y_pred)
            mod1 = getmod(y_true, y_pred)
            return tf.math.reduce_mean(loss1*mod1)
    elif mod in [3, 4]:
        @tf.function()
        def myloss(y_true, y_pred):
            loss1 = losstyp(y_true, y_pred)
            mod1 = getmod(y_true, y_pred)
            return tf.math.reduce_mean(loss1)*tf.math.reduce_mean(loss1*mod1)
    elif mod in [5, 6]:
        @tf.function()
        def myloss(y_true, y_pred):
            loss1 = losstyp(y_true, y_pred)
            mod1 = getmod(y_true, y_pred)
            return tf.math.reduce_mean(loss1**2*mod1)
    elif mod in [7]:
        @tf.function()
        def myloss(y_true, y_pred):
            return tf.math.reduce_sum((1 - y_true*y_pred)**alpha, axis=1)
            # loss1 = losstyp(y_true, y_pred)
            # return tf.math.reduce_mean(loss1**alpha)

    return myloss


def squared_hinge_mod(index=0):
    @tf.function(reduce_retracing=True)
    def loss_mod(y_true, y_pred):
        diff_use = 1.0
        y_true_adj = (y_true + 1)/2
        y_pred_adj = (y_pred + 1)/2

        # if index < 10 and index > 0:

        y_true_r = tf.math.round(y_true_adj)
        y_pred_r = tf.math.round(y_pred_adj)
        r_true = tf.math.reduce_sum(y_true_r, axis=1)
        r_pred = tf.math.reduce_sum(y_pred_r, axis=1)
        r_diff = tf.math.abs(r_true - r_pred)

        if index == 1:
            diff_use = r_diff**0.5  # #1
        elif index == 2:
            diff_use = r_diff
        elif index == 3:
            diff_use = r_diff**2
        elif index == 4:
            diff_use = (1 + r_diff)**0.5
        elif index == 5:
            diff_use = (1 + r_diff)  # #2
        elif index == 6:
            diff_use = (1 + r_diff)**2
        elif index == 7:
            diff_use = (1 + r_diff**0.5)  # #3
        elif index == 8:
            diff_use = (1 + r_diff**2)

        if index >= 10:
            # y_diff = tf.math.reduce_sum(
            #     tf.math.abs(y_true_adj - y_pred_adj), axis=1
            # )
            # TODO Next: try with tf.where
            y_diff_pre1 = tf.math.abs(y_true_adj - y_pred_adj)
            y_diff_pre2 = tf.where(
                y_diff_pre1 < 0.3333,
                tf.zeros(tf.shape(y_diff_pre1)),
                y_diff_pre1
            )
            y_diff = tf.math.reduce_sum(y_diff_pre2, axis=1)

        if index == 10:
            diff_use = y_diff**0.5  # Best, somewhat unstable
        elif index == 11:
            diff_use = y_diff  # Could be good but very unstable
        elif index == 12:
            diff_use = y_diff**2
        elif index == 13:
            diff_use = (1 + y_diff)**0.5  # Good but very unstable
        elif index == 14:
            diff_use = (1 + y_diff)
        elif index == 15:
            diff_use = (1 + y_diff)**2
        elif index == 16:
            diff_use = (1 + y_diff**0.5)
        elif index == 17:
            diff_use = (1 + y_diff**2)
        elif index == 18:
            diff_use = (y_diff*r_diff)**0.5
        elif index == 19:
            diff_use = ((y_diff*r_diff)**0.5)**0.5

        sqhng = tf.math.reduce_sum((1 - y_true*y_pred)**2, axis=1)
        return tf.math.reduce_mean(sqhng*diff_use)
    return loss_mod
