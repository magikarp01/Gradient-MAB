import SPSA
import functions

print(functions.ackley([0, 0, 0], 3))
print()

print(SPSA.gradDescent(functions.ackley, [1]*5, 5, 100, 100, .2))

#def ackley_Adjusted(x, d, a = 20, b=.2, c=2*math.pi):