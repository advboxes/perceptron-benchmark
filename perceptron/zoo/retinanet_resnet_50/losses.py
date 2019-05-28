"""losses"""
import keras
from . import backend


def focal(alpha=0.25, gamma=2.0):
    """ Create a functor for computing the focal loss.

    Parameters
    ----------
    alpha : float
        Scale the focal weight with alpha.
    gamma : float
        Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """
    def _focal(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the 
        predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Parameters
        ----------
            y_true : tensor
                Tensor of target data from the generator with shape (B, 
                N, num_classes).
            y_pred : tensor
                Tensor of predicted data from the network with shape (B, 
                N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels = y_true[:, :, :-1]
        # -1 for ignore, 0 for background, 1 for object
        anchor_state = y_true[:, :, -1]
        classification = y_pred

        # filter out "ignore" anchors
        indices = backend.where(keras.backend.not_equal(anchor_state, -1))
        labels = backend.gather_nd(labels, indices)
        classification = backend.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = backend.where(
            keras.backend.equal(
                labels,
                1),
            alpha_factor,
            1 - alpha_factor)
        focal_weight = backend.where(
            keras.backend.equal(
                labels,
                1),
            1 - classification,
            classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * \
            keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = backend.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(
            keras.backend.shape(normalizer)[0],
            keras.backend.floatx())
        normalizer = keras.backend.maximum(1.0, normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal


def smooth_l1(sigma=3.0):
    """ Create a smooth L1 loss functor.

    Parameters
    ----------
    sigma : float
        This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and 
        predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true : tensor
                Tensor from the generator of shape (B, N, 5). The last value 
                for each box is the state of the anchor (ignore, negative, 
                positive).
            y_pred : tensor
                Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices = backend.where(keras.backend.equal(anchor_state, 1))
        regression = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(
            normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1
