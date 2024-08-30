from dpipe.batch_iter import sample
from dpipe.itertools import pam


def load_by_random_id_uda(*loaders, ids_s, ids_t, weights=None, random_state=None):
    """load_by_random_id for Labelled (L) and Unlabelled (U) images loading"""
    for id_s, id_t in zip(sample(ids_s, weights, random_state), sample(ids_t, None, random_state)):
        if len(loaders) == 2:  # plain sampling:
            xs, ys = pam(loaders, id_s)
            xt, = pam(loaders[:1], id_t)
            yield xs, xt, ys
        elif len(loaders) == 3:  # tumor sampling:
            xs, ys, cs = pam(loaders, id_s)
            xt, = pam(loaders[:1], id_t)
            yield xs, xt, ys, cs
        else:
            raise TypeError(f'Only 2 (simple sampling) and 3 (tumor sampling) loaders are supported, '
                            f'but {len(loaders)} loaders given.')
