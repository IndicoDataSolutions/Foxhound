from foxhound.linear_model import LinearModel
from foxhound.utils.costs import squared_hinge, hinge

valid_cost_functions = {'hinge':hinge, 'squared_hinge':squared_hinge}

class LinearSVC(LinearModel):

    def __init__(self, cost='hinge', *args, **kwargs):

        try:
            selected_cost = valid_cost_functions[cost]
        except KeyError:
            raise ValueError("User inputed cost function '%s' not valid for %s model."
                             %(selected_cost, self.__class__.__name__))

        LinearModel.__init__(self, cost=selected_cost, **kwargs)
