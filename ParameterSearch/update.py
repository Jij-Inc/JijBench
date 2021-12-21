from jijmodeling import Problem, Element
from jijmodeling.decode import DecodedSamples, indices_combination
from jijmodeling.expression.condition import LessThanEqual, LessThan, Equal
from jijmodeling.transpilers.type_annotations import PH_VALUES_INTERFACE
from alm import ALMModel
from typing import Dict


def generator_for_pyqubo_feeddict(almmodel: ALMModel,
                                  ph_value: PH_VALUES_INTERFACE = {},
                                  mu0_value: float = 1.0) -> Dict[str, float]:
    """Generate feed dict including linear and quad term's multipliers
    Args:
        almmodel (ALMModel): alm model
        multipliers (Dict[str, float]): default multipliers.
        ph_value (PH_VALUES_INTERFACE, optional): Placeholder Value. Defaults to {}.
        mu0_value (float, optional): initial valeue of almmodel's penalty. Defaults to 1.0.
    Returns:
        Dict[str, float]: multipliers
    """
    alm_multipliers = {}
    for penalty in almmodel.penalties:
        if penalty.forall:
            forall = penalty.forall
        else:
            forall = [(Element("__dammy__", (0, 1)), None)]

        for indices in indices_combination(forall[::-1], ph_value, {}):
            index_list = [indices[i[0]] for i in forall]
            key_str = ",".join([str(i) for i in index_list])
            # muの初期値を設定する
            alm_multipliers[penalty.label + "_linear_" + key_str] = mu0_value
            alm_multipliers[penalty.label + "_quad_" + key_str] = mu0_value
    return alm_multipliers


def alm_update(almmodel: ALMModel,
               decoded: DecodedSamples,
               alm_multipliers: Dict[str, float],
               alpha: float = 3.0,
               beta: float = 1.1):
    """Update almmodel's multipliers
    Args:
        almmodel (ALMModel): alm model
        decoded (DecodedSamples): decoded answer
        alm_multipliers (Dict[str, float]): multipliers to update
        alpha (float, optional): step size  of linear term. Defaults to 3.0.
        beta (float, optional): step size of quad term. Defaults to 1.1.
        bigm_beta (float, optional): step size of bigm constraints quad term. Defaults to 0.1.
    Returns:
        Dict[str, float]: next multipliers
    """
    for constraints in decoded.constraint_expr_value:
        for const_label, const in constraints.items():
            for indices, value in const.items():
                if not indices:
                    indices = ("0")
                linear_name = const_label + "_linear_" + ",".join([str(i) for i in indices])
                quad_name = const_label + "_quad_" + ",".join([str(i) for i in indices])
                if isinstance(almmodel.problem.constraints[const_label].condition, Equal):
                    # lambdaの更新
                    alm_multipliers[linear_name] += alm_multipliers[quad_name] * value
                    # muの更新: alphaは3が良いらしい
                    alm_multipliers[quad_name] *= alpha
                elif isinstance(almmodel.problem.constraints[const_label].condition, (LessThan, LessThanEqual)):
                    alm_multipliers[linear_name] \
                            = max(0, alm_multipliers[linear_name] + alm_multipliers[quad_name] * value)
                    alm_multipliers[quad_name] *= beta
    print(alm_multipliers)
    return alm_multipliers


def make_initial_multipliers(problem: Problem):
    multipliers = {}
    for key in problem.constraints.keys():
        multipliers[key] = 1
    return multipliers


def parameter_update(problem: Problem,
                     decoded: DecodedSamples,
                     multipliers: Dict[str, float]):
    next_multipliers = {}
    for key, value in decoded.constraint_violations[0].items():
        if value > 0:
            next_multipliers[key] = 5 * multipliers[key]
        else:
            next_multipliers[key] = multipliers[key]
    return next_multipliers
