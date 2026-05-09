import math

def f(x):
    return 2 * x * math.cos(x) - (x - 2)**2

def df(x):
    return 2 * math.cos(x) - 2 * x * math.sin(x) - 2 * (x - 2)

print(f"Bolzano:")
print(f"f(0.8) = {f(0.8):.6f}")
print(f"f(1.4) = {f(1.4):.6f}")

def g1(x):
    val = 2 * x * math.cos(x)
    if val < 0: return None
    return 2 - math.sqrt(val)

def g2(x):
    return 4 / (2 * math.cos(x) - x + 4)

print("\nTesting g functions for x0=1:")
x0 = 1.0
print(f"g1(1) = {g1(x0)}, f(g1(1)) = {f(g1(x0)) if g1(x0) else 'None'}")
print(f"g2(1) = {g2(x0)}, f(g2(1)) = {f(g2(1))}")

val = 2 * 1 * math.cos(1)
g1_prime = -0.5 * (1/math.sqrt(val)) * (2 * math.cos(1) - 2 * math.sin(1))
print(f"g1'(1) = {g1_prime:.6f}") 

denom = 2 * math.cos(1) - 1 + 4
g2_prime = -4 * (-2 * math.sin(1) - 1) / denom**2
print(f"g2'(1) = {g2_prime:.6f}") 

def steffensen(g, x0, tol=1e-6, max_iter=20):
    print(f"\nSteffensen-Aitken with x0={x0}")
    print(f"{'Iter':<5} {'p0':<12} {'p1':<12} {'p2':<12} {'p':<12} {'error':<12}")
    p0 = x0
    for i in range(max_iter):
        p1 = g(p0)
        p2 = g(p1)
        denominator = p2 - 2*p1 + p0
        if abs(denominator) < 1e-15:
            print("Denominator zero")
            break
        p = p0 - ((p1 - p0)**2) / denominator
        error = abs(p - p0)
        print(f"{i:<5} {p0:<12.6f} {p1:<12.6f} {p2:<12.6f} {p:<12.6f} {error:<12.6f}")
        if error < tol:
            return p
        p0 = p
    return p0

root_a = steffensen(g1, 1.0)

def newton_raphson(f, df, x0, tol=1e-8, max_iter=20):
    print(f"\nNewton-Raphson with x0={x0}")
    print(f"{'Iter':<5} {'xi':<15} {'f(xi)':<15} {'df(xi)':<15} {'xi+1':<15} {'error':<15}")
    xi = x0
    for i in range(max_iter):
        f_xi = f(xi)
        df_xi = df(xi)
        if abs(df_xi) < 1e-15:
            print("Derivative zero")
            break
        xi_plus_1 = xi - f_xi / df_xi
        error = abs(xi_plus_1 - xi)
        print(f"{i:<5} {xi:<15.8f} {f_xi:<15.8f} {df_xi:<15.8f} {xi_plus_1:<15.8f} {error:<15.8f}")
        if error < tol:
            return xi_plus_1
        xi = xi_plus_1
    return xi

root_b = newton_raphson(f, df, 1.0)
