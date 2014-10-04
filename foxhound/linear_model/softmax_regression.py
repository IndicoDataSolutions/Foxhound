from foxhound.linear_model import LogisticRegression
from foxhound.utils.costs import CCE

class SoftmaxRegression(LogisticRegression):

	def cost(self):
		return CCE(self.Y, self.pred)
