#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef double (*func_t)(double);

typedef struct {
    int iter;
    double a;
    double b;
    double c;
    double fc;
    double error;
} IterBiseccion;

typedef struct {
    int iter;
    double x_n;
    double x_next;
    double error;
    double fx_next;
    bool tiene_fx;
} IterPuntoFijo;

typedef struct {
    const char *nombre;
    func_t f;
    func_t g;
} Sistema;

static double f1(double x) { return x * x * x - x - 2.0; }
static double g1(double x) { return cbrt(x + 2.0); }

static double f2(double x) { return cos(x) - x; }
static double g2(double x) { return cos(x); }

static double f3(double x) { return x * x - 2.0; }
static double g3(double x) { return 0.5 * (x + 2.0 / x); }

static const Sistema SISTEMAS[] = {
    {"f(x)=x^3-x-2      | g(x)=cuberoot(x+2)", f1, g1},
    {"f(x)=cos(x)-x     | g(x)=cos(x)", f2, g2},
    {"f(x)=x^2-2        | g(x)=0.5*(x+2/x)", f3, g3},
};

static const int NUM_SISTEMAS = (int)(sizeof(SISTEMAS) / sizeof(SISTEMAS[0]));

static void limpiar_entrada(void) {
    int c;
    while ((c = getchar()) != '\n' && c != EOF) {
    }
}

static int leer_entero(const char *prompt, int min_val) {
    int v;
    while (1) {
        printf("%s", prompt);
        if (scanf("%d", &v) == 1 && v >= min_val) {
            limpiar_entrada();
            return v;
        }
        printf("Entrada invalida. Intenta de nuevo.\n");
        limpiar_entrada();
    }
}

static double leer_double(const char *prompt) {
    double v;
    while (1) {
        printf("%s", prompt);
        if (scanf("%lf", &v) == 1) {
            limpiar_entrada();
            return v;
        }
        printf("Entrada invalida. Intenta de nuevo.\n");
        limpiar_entrada();
    }
}

static int elegir_sistema(void) {
    printf("\nElige un sistema:\n");
    for (int i = 0; i < NUM_SISTEMAS; i++) {
        printf("  %d) %s\n", i + 1, SISTEMAS[i].nombre);
    }
    return leer_entero("Opcion: ", 1) - 1;
}

static int biseccion(
    func_t f,
    double a,
    double b,
    double tol,
    int max_iter,
    IterBiseccion *historial,
    int *n_historial,
    double *raiz
) {
    double fa = f(a);
    double fb = f(b);
    if (!isfinite(fa) || !isfinite(fb)) {
        return -2;
    }
    if (fa * fb > 0.0) {
        return -1;
    }

    for (int i = 1; i <= max_iter; i++) {
        double c = 0.5 * (a + b);
        double fc = f(c);
        double error = 0.5 * (b - a);
        if (!isfinite(fc)) {
            return -2;
        }

        historial[i - 1].iter = i;
        historial[i - 1].a = a;
        historial[i - 1].b = b;
        historial[i - 1].c = c;
        historial[i - 1].fc = fc;
        historial[i - 1].error = fabs(error);

        if (fabs(fc) < tol || fabs(error) < tol) {
            *n_historial = i;
            *raiz = c;
            return 0;
        }

        if (fa * fc < 0.0) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }

    *n_historial = max_iter;
    *raiz = 0.5 * (a + b);
    return 1;
}

static int punto_fijo(
    func_t g,
    func_t f,
    double x0,
    double tol,
    int max_iter,
    IterPuntoFijo *historial,
    int *n_historial,
    double *raiz
) {
    double xn = x0;
    if (!isfinite(xn)) {
        return -2;
    }

    for (int i = 1; i <= max_iter; i++) {
        double x_next = g(xn);
        if (!isfinite(x_next)) {
            return -2;
        }
        double error = fabs(x_next - xn);

        historial[i - 1].iter = i;
        historial[i - 1].x_n = xn;
        historial[i - 1].x_next = x_next;
        historial[i - 1].error = error;

        if (f != NULL) {
            double fx = f(x_next);
            historial[i - 1].fx_next = fx;
            historial[i - 1].tiene_fx = isfinite(fx);
        } else {
            historial[i - 1].fx_next = NAN;
            historial[i - 1].tiene_fx = false;
        }

        if (error < tol) {
            *n_historial = i;
            *raiz = x_next;
            return 0;
        }

        xn = x_next;
    }

    *n_historial = max_iter;
    *raiz = xn;
    return 1;
}

static void imprimir_tabla_biseccion(const IterBiseccion *h, int n) {
    printf("\nTabla de iteraciones (Biseccion)\n");
    printf("Iter |            a |            b |            c |         f(c) |        error\n");
    printf("-----+--------------+--------------+--------------+--------------+-------------\n");
    for (int i = 0; i < n; i++) {
        printf(
            "%4d | %12.8f | %12.8f | %12.8f | %12.4e | %11.4e\n",
            h[i].iter,
            h[i].a,
            h[i].b,
            h[i].c,
            h[i].fc,
            h[i].error
        );
    }
}

static void imprimir_tabla_punto_fijo(const IterPuntoFijo *h, int n) {
    printf("\nTabla de iteraciones (Punto Fijo)\n");
    printf("Iter |          x_n |        x_n+1 |        error |      f(x_n+1)\n");
    printf("-----+--------------+--------------+--------------+--------------\n");
    for (int i = 0; i < n; i++) {
        if (h[i].tiene_fx) {
            printf(
                "%4d | %12.8f | %12.8f | %12.4e | %12.4e\n",
                h[i].iter,
                h[i].x_n,
                h[i].x_next,
                h[i].error,
                h[i].fx_next
            );
        } else {
            printf(
                "%4d | %12.8f | %12.8f | %12.4e | %12s\n",
                h[i].iter,
                h[i].x_n,
                h[i].x_next,
                h[i].error,
                "N/A"
            );
        }
    }
}

static void graficar_serie_ascii(const char *titulo, const double *y, int n) {
    const int ancho = 60;
    const int alto = 14;
    char canvas[14][61];

    if (n <= 0) {
        return;
    }

    double min_y = y[0];
    double max_y = y[0];
    for (int i = 1; i < n; i++) {
        if (y[i] < min_y) min_y = y[i];
        if (y[i] > max_y) max_y = y[i];
    }
    double rango = max_y - min_y;
    if (rango == 0.0) {
        rango = 1.0;
    }

    for (int r = 0; r < alto; r++) {
        for (int c = 0; c < ancho; c++) {
            canvas[r][c] = ' ';
        }
        canvas[r][ancho] = '\0';
    }

    for (int c = 0; c < ancho; c++) {
        int idx = (n == 1) ? 0 : (int)llround((double)c * (double)(n - 1) / (double)(ancho - 1));
        double normalizado = (y[idx] - min_y) / rango;
        int r = (int)llround((1.0 - normalizado) * (alto - 1));
        if (r < 0) r = 0;
        if (r >= alto) r = alto - 1;
        canvas[r][c] = '*';
    }

    printf("\n%s\n", titulo);
    for (int r = 0; r < alto; r++) {
        if (r == 0) {
            printf("%10.4e | %s\n", max_y, canvas[r]);
        } else if (r == alto - 1) {
            printf("%10.4e | %s\n", min_y, canvas[r]);
        } else {
            printf("           | %s\n", canvas[r]);
        }
    }
    printf("           +");
    for (int c = 0; c < ancho + 2; c++) {
        putchar('-');
    }
    printf("\n           iter 1");
    for (int c = 0; c < ancho - 12; c++) {
        putchar(' ');
    }
    printf("iter %d\n", n);
}

static void ejecutar_biseccion(void) {
    int idx = elegir_sistema();
    func_t f = SISTEMAS[idx].f;

    double a = leer_double("a: ");
    double b = leer_double("b: ");
    double tol = leer_double("tolerancia (ej. 1e-7): ");
    int max_iter = leer_entero("max_iter: ", 1);

    IterBiseccion *historial = (IterBiseccion *)malloc((size_t)max_iter * sizeof(IterBiseccion));
    if (historial == NULL) {
        printf("No hay memoria suficiente.\n");
        return;
    }

    int n_hist = 0;
    double raiz = NAN;
    int estado = biseccion(f, a, b, tol, max_iter, historial, &n_hist, &raiz);

    if (estado == -1) {
        printf("\nError: f(a) y f(b) tienen el mismo signo.\n");
        free(historial);
        return;
    }
    if (estado == -2) {
        printf("\nError: se encontro un valor invalido (NaN/Inf).\n");
        free(historial);
        return;
    }

    imprimir_tabla_biseccion(historial, n_hist);
    printf("\nRaiz aproximada: %.12f\n", raiz);
    printf("Iteraciones: %d\n", n_hist);
    printf("|f(raiz)|: %.4e\n", fabs(f(raiz)));
    if (estado == 1) {
        printf("Aviso: se alcanzo max_iter sin cumplir la tolerancia.\n");
    }

    double *serie_c = (double *)malloc((size_t)n_hist * sizeof(double));
    double *serie_error = (double *)malloc((size_t)n_hist * sizeof(double));
    if (serie_c != NULL && serie_error != NULL) {
        for (int i = 0; i < n_hist; i++) {
            serie_c[i] = historial[i].c;
            serie_error[i] = historial[i].error;
        }
        graficar_serie_ascii("Grafica ASCII: convergencia de c", serie_c, n_hist);
        graficar_serie_ascii("Grafica ASCII: error por iteracion", serie_error, n_hist);
    }
    free(serie_c);
    free(serie_error);
    free(historial);
}

static void ejecutar_punto_fijo(void) {
    int idx = elegir_sistema();
    func_t f = SISTEMAS[idx].f;
    func_t g = SISTEMAS[idx].g;

    double x0 = leer_double("x0: ");
    double tol = leer_double("tolerancia (ej. 1e-7): ");
    int max_iter = leer_entero("max_iter: ", 1);

    IterPuntoFijo *historial = (IterPuntoFijo *)malloc((size_t)max_iter * sizeof(IterPuntoFijo));
    if (historial == NULL) {
        printf("No hay memoria suficiente.\n");
        return;
    }

    int n_hist = 0;
    double raiz = NAN;
    int estado = punto_fijo(g, f, x0, tol, max_iter, historial, &n_hist, &raiz);

    if (estado == -2) {
        printf("\nError: g(x) produjo un valor invalido (NaN/Inf).\n");
        free(historial);
        return;
    }

    imprimir_tabla_punto_fijo(historial, n_hist);
    printf("\nPunto fijo aproximado: %.12f\n", raiz);
    printf("Iteraciones: %d\n", n_hist);
    printf("|f(raiz)|: %.4e\n", fabs(f(raiz)));
    printf("|g(raiz)-raiz|: %.4e\n", fabs(g(raiz) - raiz));
    if (estado == 1) {
        printf("Aviso: se alcanzo max_iter sin cumplir la tolerancia.\n");
    }

    double *serie_x = (double *)malloc((size_t)n_hist * sizeof(double));
    double *serie_error = (double *)malloc((size_t)n_hist * sizeof(double));
    if (serie_x != NULL && serie_error != NULL) {
        for (int i = 0; i < n_hist; i++) {
            serie_x[i] = historial[i].x_next;
            serie_error[i] = historial[i].error;
        }
        graficar_serie_ascii("Grafica ASCII: convergencia de x_n+1", serie_x, n_hist);
        graficar_serie_ascii("Grafica ASCII: error por iteracion", serie_error, n_hist);
    }
    free(serie_x);
    free(serie_error);
    free(historial);
}

int main(void) {
    printf("==============================================\n");
    printf(" Dashboard C - Metodos Numericos (Local)\n");
    printf("==============================================\n");

    while (1) {
        printf("\nMenu principal\n");
        printf("  1) Biseccion\n");
        printf("  2) Punto Fijo\n");
        printf("  0) Salir\n");

        int opcion = leer_entero("Elige opcion: ", 0);
        if (opcion == 0) {
            printf("Hasta luego.\n");
            break;
        } else if (opcion == 1) {
            ejecutar_biseccion();
        } else if (opcion == 2) {
            ejecutar_punto_fijo();
        } else {
            printf("Opcion no valida.\n");
        }
    }

    return 0;
}
