import math
from sympy import symbols, diff

x = symbols("x")

f = 69*x**4-45*x**3-234
f_prime = diff(f, x)  

x_inc = 1.0

tolerance = 1e-10
max_iterations = 1000
iteration = 0

while abs(f.subs(x, x_inc)) > tolerance and iteration < max_iterations:
    f_val = f.subs(x, x_inc)
    f_prime_val = f_prime.subs(x, x_inc)
    
    x_inc = x_inc - f_val / f_prime_val
    iteration += 1
    
    print(f"Iteration {iteration}: x = {round(x_inc,6)}, f(x) = {round(f.subs(x, x_inc), 6)}")

print(f"\nRoot found: x = {x_inc}")
print(f"Function value at root: f({x_inc}) = {f.subs(x, x_inc)}")
print(f"Converged in {iteration} iterations")