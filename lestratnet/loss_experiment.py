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


def loss_setup2(loss='categorical_crossentropy', mod=None, alpha=1.0, cep=None):
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
            y_true_adj = (y_true + 1)/2

            y_true_r = tf.math.round(y_true_adj)

            r_true = tf.math.reduce_sum(y_true_r, axis=1)
            return tf.cast(1.0, tf.float32), r_true
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

            return r_diff**alpha, r_true

        if mod in [1, 3, 5]:
            @tf.function()
            def getmod(y_true, y_pred):
                return getrdiff(y_true, y_pred)
        elif mod in [2, 4, 6]:
            @tf.function()
            def getmod(y_true, y_pred):
                grd1, grd2 = getrdiff(y_true, y_pred)
                return 1 + grd1, grd2

    if mod in [0, 1, 2] or mod is None:
        @tf.function()
        def myloss(y_true, y_pred):
            loss1 = losstyp(y_true, y_pred)
            mod1, r_true = getmod(y_true, y_pred)
            lossprod = loss1*mod1

            shw_sum = tf.cast(0.0, tf.float32)
            for k in range(int(tf.math.reduce_max(r_true)) + 1):
                ireg = tf.cast(k, tf.float32)
                shw_sum += tf.math.reduce_mean(
                    lossprod[r_true == ireg]
                )

            return shw_sum
    elif mod in [3, 4]:
        @tf.function()
        def myloss(y_true, y_pred):
            loss1 = losstyp(y_true, y_pred)
            mod1, r_true = getmod(y_true, y_pred)
            lossprod = loss1*mod1

            shw_sum = tf.cast(0.0, tf.float32)
            for k in range(int(tf.math.reduce_max(r_true)) + 1):
                ireg = tf.cast(k, tf.float32)
                shw_sum += tf.math.reduce_mean(
                    lossprod[r_true == ireg]
                )

            return tf.math.reduce_mean(loss1)*shw_sum
    elif mod in [5, 6]:
        @tf.function()
        def myloss(y_true, y_pred):
            loss1 = losstyp(y_true, y_pred)
            mod1, r_true = getmod(y_true, y_pred)
            lossprod = loss1**2*mod1

            shw_sum = tf.cast(0.0, tf.float32)
            for k in range(int(tf.math.reduce_max(r_true)) + 1):
                ireg = tf.cast(k, tf.float32)
                shw_sum += tf.math.reduce_mean(
                    lossprod[r_true == ireg]
                )

            return shw_sum
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


##############################################################################
# OLD ATTEMPTS AT A LOSS FUNCTION. KEPT HERE AS REFERENCE BUT WILL BE REMOVED
def my_loss_prev(
    name="binary_crossentropy",
    jumping=None
):
    if name == "binary_crossentropy":
        first_loss = tf.keras.losses.BinaryCrossentropy()

        def get_rdiff(y_true, y_pred):
            y_true_r = tf.math.round(y_true)
            y_pred_r = tf.math.round(y_pred)
            r_true = tf.math.reduce_sum(y_true_r, axis=1)
            r_pred = tf.math.reduce_sum(y_pred_r, axis=1)
            r_diff = r_true - r_pred
            return r_diff

    elif name == "squared_hinge":
        first_loss = tf.keras.losses.SquaredHinge()

        def get_rdiff(y_true, y_pred):
            y_true_r = tf.math.round((y_true + 1)/2)
            y_pred_r = tf.math.round((y_pred + 1)/2)
            r_true = tf.math.reduce_sum(y_true_r, axis=1)
            r_pred = tf.math.reduce_sum(y_pred_r, axis=1)
            r_diff = r_true - r_pred
            return r_diff

    elif name == "categorical_crossentropy":
        first_loss = tf.keras.losses.CategoricalCrossentropy()

        def get_rdiff(y_true, y_pred):
            r_true = tf.cast(tf.math.argmax(y_true, axis=1), y_pred.dtype)
            r_pred = tf.cast(tf.math.argmax(y_pred, axis=1), y_pred.dtype)
            r_diff = r_true - r_pred
            return r_diff
    else:
        raise ValueError(
            "Only `binary_crossentropy`, `squared_hinge` and "
            + "`categorical_crossentropy` have been configured "
            + "to use by `name`."
        )

    def _my_loss(y_true, y_pred):
        firstl = first_loss(y_true, y_pred)
        if jumping == "squared":
            rdiff = get_rdiff(y_true, y_pred)
            res = tf.math.reduce_mean(rdiff**2)
        elif jumping == "absolute":
            rdiff = get_rdiff(y_true, y_pred)
            res = tf.math.reduce_mean(tf.math.abs(rdiff))
        else:
            res = 1.0

        return firstl*res

    return _my_loss


def my_loss(
    name="binary_crossentropy",
    jumping=None
):
    if name == "binary_crossentropy":
        first_loss = tf.keras.losses.BinaryCrossentropy()

        def get_rdiff(y_true, y_pred):
            y_true_r = tf.math.round(y_true)
            y_pred_r = tf.math.round(y_pred)
            r_true = tf.math.reduce_sum(y_true_r, axis=1)
            r_pred = tf.math.reduce_sum(y_pred_r, axis=1)
            r_diff = r_true - r_pred
            return r_diff, r_pred

    elif name == "squared_hinge":
        first_loss = tf.keras.losses.SquaredHinge()

        def get_rdiff(y_true, y_pred):
            y_true_cor = (y_true + 1)/2
            y_pred_cor = (y_pred + 1)/2
            # y_true_r = tf.math.round(y_true_cor)
            y_pred_r = tf.math.round(y_pred_cor)
            # r_true = tf.math.reduce_sum(y_true_r, axis=1)
            r_pred = tf.math.reduce_sum(y_pred_r, axis=1)
            r_true_cor = tf.math.reduce_sum(y_true_cor, axis=1)
            r_pred_cor = tf.math.reduce_sum(y_pred_cor, axis=1)
            y_diff = tf.math.reduce_sum(
                tf.math.abs(y_true_cor - y_pred_cor), axis=1
            )
            # r_diff = r_true - r_pred
            r_diff = r_true_cor - r_pred_cor
            return tf.math.sqrt(tf.math.abs(y_diff*r_diff)), r_pred
            # return y_diff, r_pred

    elif name == "categorical_crossentropy":
        first_loss = tf.keras.losses.CategoricalCrossentropy()

        def get_rdiff(y_true, y_pred):
            r_true_i = tf.math.argmax(y_true, axis=1)
            r_pred_i = tf.math.argmax(y_pred, axis=1)
            r_true = tf.cast(r_true_i, y_pred.dtype)
            r_pred = tf.cast(r_pred_i, y_pred.dtype)

            # ================
            y_pred_f = tf.reshape(y_pred, [-1])
            yps = tf.shape(y_pred)
            r_true_if = r_true_i + tf.cast(
                tf.range(yps[0])*yps[1], r_true_i.dtype
            )
            y_pmft = tf.gather(y_pred_f, r_true_if)
            ydiffs = tf.ones(tf.shape(y_pmft)) - y_pmft
            # ================
            r_diff = r_true - r_pred
            return r_diff*ydiffs, r_pred
    else:
        raise ValueError(
            "Only `binary_crossentropy`, `squared_hinge` and "
            + "`categorical_crossentropy` have been configured "
            + "to use by `name`."
        )

    # TODO Check for the advantage of tf.function decorator
    if jumping == "squared":
        @tf.function(reduce_retracing=True)
        def _my_loss(y_true, y_pred):
            firstl = first_loss(y_true, y_pred)
            rdiff, rpred = get_rdiff(y_true, y_pred)
            res = tf.math.reduce_mean((rdiff)**2)
            return firstl*(1.0 + res)
    elif jumping == "absolute":
        @tf.function(reduce_retracing=True)
        def _my_loss(y_true, y_pred):
            firstl = first_loss(y_true, y_pred)
            rdiff, rpred = get_rdiff(y_true, y_pred)
            res = tf.math.reduce_mean(tf.math.abs(rdiff))
            return firstl*(1.0 + res)
    else:
        @tf.function(reduce_retracing=True)
        def _my_loss(y_true, y_pred):
            return first_loss(y_true, y_pred)
    # ===================================
    # @tf.function(reduce_retracing=True)
    # def _my_loss(y_true, y_pred):
    #     # global weights_as_tensor
    #     firstl = first_loss(y_true, y_pred)
    #     if jumping == "squared":
    #         rdiff, rpred = get_rdiff(y_true, y_pred)
    #         res = tf.math.reduce_mean((rdiff)**2)
    #         # maxrdiff = tf.math.reduce_max(tf.math.abs(rdiff))
    #     elif jumping == "absolute":
    #         rdiff, rpred = get_rdiff(y_true, y_pred)
    #         res = tf.math.reduce_mean(tf.math.abs(rdiff))
    #         # maxrdiff = tf.math.reduce_max(tf.math.abs(rdiff))
    #     else:
    #         res = tf.constant(1.0)
    #         # maxrdiff = tf.constant(1.0)

    #     # tf.print(maxrdiff, res)

    #     # tf.print(maxrdiff, res, stdrdiff, firstl*(0.001 + res))
    #     return firstl*(1.0 + res)

    return _my_loss
