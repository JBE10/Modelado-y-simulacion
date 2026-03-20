#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef double (*func_t)(double);

double biseccion(func_t f, double a, double b, double tol, int max_iter, int *iteraciones) {
    double fa = f(a);
    double fb = f(b);

    if (fa * fb > 0.0) {
        fprintf(stderr, "Error: el intervalo [a, b] no encierra una raiz.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 1; i <= max_iter; i++) {
        double c = (a + b) / 2.0;
        double fc = f(c);

        if (fabs(fc) < tol || (b - a) / 2.0 < tol) {
            if (iteraciones != NULL) {
                *iteraciones = i;
            }
            return c;
        }

        if (fa * fc < 0.0) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }

    if (iteraciones != NULL) {
        *iteraciones = max_iter;
    }
    return (a + b) / 2.0;
}

double f(double x) {
    return x * x * x - x - 2.0; /* Raiz real aproximada: 1.52138 */
}

int main(void) {
    int iters = 0;
    double raiz = biseccion(f, 1.0, 2.0, 1e-10, 200, &iters);

    printf("Raiz aproximada: %.12f\n", raiz);
    printf("Iteraciones: %d\n", iters);
    printf("f(raiz): %.12e\n", f(raiz));

    return 0;
}
