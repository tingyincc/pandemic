# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
import abc
import dataclasses
from typing import Any, cast, Type, TypeVar, Dict, List

import istype
import numpy as np

__all__ = ['required', 'abstract_class_property', 'checked_cast', 'shallow_asdict', 'cluster_into_random_sized_groups',
           'integer_partitions']

_T = TypeVar('_T')


def get_compliance_prob(init_prob, day, cur_stage = 0)->int:
    decay_function="custom"

    cal_prob = init_prob

    if decay_function == "poly":
        cal_prob = init_prob - 0.0001 * pow( day, 2)
    elif decay_function == "poly_increase":
        cal_prob = np.log10(0.1*day+1)
    elif decay_function == "linear":
        cal_prob = init_prob - 0.01 * day
    elif decay_function == "linear_increase":
        cal_prob = 0.01 * day
    elif decay_function == "exp":
        cal_prob = init_prob * 0.95 ** day
    elif decay_function == "exp_increase":
        cal_prob =  (1.05 ** day) /100
    elif decay_function == "0":
        cal_prob = 0.01
    elif decay_function == "custom":
        # if day < 25:
        #     cal_prob = (1.211 ** day) /100
        # else:
        #     cal_prob = 0.989 - 0.0001 * pow( day-25, 2)
        if day < 25:
            cal_prob = np.log10(0.36*day+1)
        else:
            cal_prob = 0.984 - 0.0001 * pow( day-25, 2)

        # if day < 25:
        #     cal_prob = 0.7 + 0.011*day
        # else:
        #     cal_prob = 0.964 - 0.0001 * pow( day-25, 2)
        # if day < 25:
        #     cal_prob = 0.7 + 0.011*day
        # else:
        #     cal_prob = 0.964 - 0.007*(day-25)
        # if day < 60:
        #     cal_prob = 0.3 + 0.011*day
        # else:
        #     cal_prob = 0.96 - 0.007*(day-59)
        # if day < 40:
        #     return 0.7 + 0.007*day
        # else:
        #     return 0.99 - 0.01*(day-39)

    elif decay_function == "stage":

        cal_prob = init_prob - 0.01 * day * cur_stage

    cal_prob = cal_prob if cal_prob >= 0 else 0
    cal_prob = cal_prob if cal_prob <= 1 else 1

    return cal_prob

def required() -> _T:
    def required_err() -> Any:
        raise ValueError('Missing required field')

    return cast(_T, dataclasses.field(default_factory=required_err))


def abstract_class_property() -> _T:
    @abc.abstractmethod  # type: ignore
    def inner() -> Any:
        raise NotImplementedError

    return cast(_T, inner)


def checked_cast(type: Type[_T], obj: Any) -> _T:
    """
    Method for executing a safe cast in python
    """
    assert istype.isinstanceof(obj, type)
    return obj  # type: ignore


def shallow_asdict(x: Any) -> Dict[str, Any]:
    assert dataclasses.is_dataclass(x)
    return {field.name: getattr(x, field.name) for field in dataclasses.fields(x)}


def cluster_into_random_sized_groups(orig_list: List[int],
                                     min_group_size: int,
                                     max_group_size: int,
                                     numpy_rng: np.random.RandomState) -> List[List[int]]:
    final_list = []
    cnt = 0
    while cnt < len(orig_list):
        size = numpy_rng.randint(min_group_size, max_group_size + 1)
        final_list.append(orig_list[cnt: cnt + size])
        cnt += size
    return final_list


def integer_partitions(x: int, n_partitions: int) -> List[int]:
    _x = x // n_partitions
    return [_x + 1 if i < x % n_partitions else _x for i in range(n_partitions)]
