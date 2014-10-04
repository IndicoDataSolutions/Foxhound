from linear_model import LogisticRegression
from utils.cost import CCE

class SoftmaxRegression(LogisticRegression):

	def cost(self):
		return CCE(self.Y, self.pred)
