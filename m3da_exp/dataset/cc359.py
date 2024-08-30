from connectome import Transform, Output
from deli import load_json

from ..paths import REPO_PATH


class BuildMask(Transform):
    __inherit__ = True

    def mask(wm_gm_csf):
        return wm_gm_csf


class AddFold(Transform):
    __inherit__ = True
    _cc359_wmgmcsf_split: tuple = tuple((k, tuple(map(tuple, vs))) for k, vs in
                                        load_json(REPO_PATH / 'dataset' / 'cc359_wmgmcsf_split.json').items())

    def domain(field, vendor):
        return f'{field}-{vendor}'

    def fold(id, domain: Output, _cc359_wmgmcsf_split):
        split = 'tr' if id in dict(_cc359_wmgmcsf_split)[domain][0] else 'ts'
        return f'{domain}|{split}'
