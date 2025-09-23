from dataclasses import dataclass
#from io import valid_seek_flags
from typing import Any, Iterable, List, Tuple, Protocol
from collections import deque, defaultdict
# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_1 = list(vals)
    vals_2 = list(vals)

    vals_1[arg] += epsilon
    vals_2[arg] -= epsilon

    return (f(*vals_1) - f(*vals_2)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visit_list = set()
    res = []

    def dfs(node: Variable) -> Any:
        if node.unique_id in visit_list or node.is_constant():
            return
        if not node.is_leaf():
            for v in node.parents:
                dfs(v)
        visit_list.add(node.unique_id)
        res.append(node)

    dfs(variable)
    return reversed(res)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    top_sort = topological_sort(variable)

    vec = defaultdict(float)
    vec[variable.unique_id] = deriv

    for node in top_sort:
        if node.is_leaf():
            continue
        if node.unique_id in vec.keys():
            deriv = vec[node.unique_id]
        deriv = node.chain_rule(deriv)
        for key, item in deriv:
            if key.is_leaf():
                key.accumulate_derivative(item)
            else:
                vec[key.unique_id] += item


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
