from typing import List, Tuple, Type
import pyqubo
from jijmodeling.decode import DecodedSamples, indices_combination
from jijmodeling.expression.condition import Condition
from jijmodeling.expression.expression import Expression
from jijmodeling.expression.variables.variable import Element
from jijmodeling.problem import Problem, ProblemKind
from jijmodeling.transpilers.elementwise_substitution import validate_inputdata
from jijmodeling.transpilers.to_pyqubo import convert_to_pyqubo, PyQUBODeciVarGenerator
from jijmodeling.transpilers.type_annotations import PH_VALUES_INTERFACE, FIXED_VARIABLES


class ALMPenalty:
    def __init__(self,
                 label: str,
                 cond_kind: Type,
                 linear: Expression,
                 quad: Expression,
                 forall: List[Tuple[Element, Condition]] = []) -> None:
        """
        Args:
            label (str): label of constraint
            cond_kind (Type): condition of constraints
            linear (Expression): linear penalty term of constraint
            quad (Expression): quad penalty term of constraint
            forall (List[Tuple[Element, Condition]]): forall condition of constraint
        """
        self.label = label
        self.cond_kind = cond_kind
        self.linear = linear
        self.quad = quad
        self.forall = forall


class ALMModel:
    def __init__(self,
                 problem: Problem,
                 cost: Expression,
                 penalties: List[ALMPenalty] = []) -> None:
        """
        Args:
            problem (Problem): Original Optimization Problem
            cost (Expression): Cost term of oroginal optimization problem
            penalties (List[ALMPenalty], optional): linear and quad term of problem's constraints. Defaults to [].
        """
        self.problem = problem
        self.cost = cost
        self.penalties = penalties


def alm_transpile(problem: Problem) -> ALMModel:
    """ Transpile original problem to alm model
    Args:
        problem (Problem): problem including cost, penalty and constraints
    Returns:
        ALMModel: transpiled to alm model
    """
    if problem.kind == ProblemKind.Maximum:
        cost = -1 * problem.objective
    else:
        cost = problem.objective

    penalties = []
    for label, constraint in problem.constraints.items():
        linear = constraint.condition.left - constraint.condition.right
        quad = linear**2
        penalty = ALMPenalty(label, constraint.condition.__class__, linear, quad, constraint.forall)
        penalties.append(penalty)
    return ALMModel(problem, cost, penalties)


def alm_pyqubo_compile(almmodel: ALMModel,
                       ph_value: PH_VALUES_INTERFACE = {},
                       fixed_variables: FIXED_VARIABLES = {}):
    """compile alm model to pyqubo object
    Args:
        almmodel (ALMModel): alm model.
        ph_value (PH_VALUES_INTERFACE, optional): Placeholder Value. Defaults to {}.
        fixed_variables (FIXED_VARIABLES, optional): fixed variables. Defaults to {}.
    Returns:
        pyqbuo: pyqubo object
    """
    data_ph_value, _, data_fixed_var = validate_inputdata(almmodel.problem, ph_value, {}, fixed_variables)
    pyq_deci_vars = PyQUBODeciVarGenerator(data_fixed_var)
    pyq_model = almmodel.cost.to_pyqubo(data_ph_value, data_fixed_var)

    for penalty in almmodel.penalties:
        if penalty.forall:
            forall = penalty.forall
        else:
            forall = [(Element("__dammy__", (0, 1)), None)]
        for indices in indices_combination(forall[::-1], ph_value, {}):
            index_list = [indices[i[0]] for i in forall]
            key_str = ",".join([str(i) for i in index_list])
            lam_i = pyqubo.Placeholder(penalty.label + "_linear_" + key_str)
            linear = lam_i * convert_to_pyqubo(penalty.linear, data_ph_value, pyq_deci_vars, indices)
            mu_i = pyqubo.Placeholder(penalty.label + "_quad_" + key_str)
            quad = mu_i / 2 * convert_to_pyqubo(penalty.quad, data_ph_value, pyq_deci_vars, indices)
            pyq_model += linear + quad

    return pyq_model


def feasible(decoded: DecodedSamples):
    """Check whether solution is feasible.
    Args:
        decoded (DecodedSamples): decoded answer
    Returns:
        Bool: if decoded is feasible, return True
    """
    for constraints in decoded.constraint_violations:
        if not all(value == 0.0 for key, value in constraints.items()):
            return False
    return True
