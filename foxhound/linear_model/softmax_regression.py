from foxhound.linear_model import LogisticRegression
from foxhound.utils.costs import cce

class SoftmaxRegression(LogisticRegression):

	def cost(self):
		return cce(self.Y, self.pred)
