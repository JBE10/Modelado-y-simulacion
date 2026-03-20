#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef double (*func_t)(double);

typedef struct {
    const char *nombre;
    func_t f;
    func_t g;
} Sistema;

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
} IterPuntoFijo;

static double f1(double x) { return x * x * x - x - 2.0; }
static double g1(double x) { return cbrt(x + 2.0); }

static double f2(double x) { return cos(x) - x; }
static double g2(double x) { return cos(x); }

static double f3(double x) { return x * x - 2.0; }
static double g3(double x) { return 0.5 * (x + 2.0 / x); }

static const Sistema SISTEMAS[] = {
    {"x^3-x-2", f1, g1},
    {"cos(x)-x", f2, g2},
    {"x^2-2", f3, g3},
};

static const int NUM_SISTEMAS = (int)(sizeof(SISTEMAS) / sizeof(SISTEMAS[0]));

static void print_error_json(const char *msg) {
    printf("{\"ok\":false,\"error\":\"%s\"}\n", msg);
}

static bool parse_double(const char *s, double *out) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (s == end || *end != '\0' || !isfinite(v)) {
        return false;
    }
    *out = v;
    return true;
}

static bool parse_int(const char *s, int *out) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s == end || *end != '\0' || v < 1 || v > 1000000) {
        return false;
    }
    *out = (int)v;
    return true;
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
        double error = 0.5 * fabs(b - a);
        if (!isfinite(fc)) {
            return -2;
        }

        historial[i - 1].iter = i;
        historial[i - 1].a = a;
        historial[i - 1].b = b;
        historial[i - 1].c = c;
        historial[i - 1].fc = fc;
        historial[i - 1].error = error;

        if (fabs(fc) < tol || error < tol) {
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
        double fx = f(x_next);
        if (!isfinite(fx)) {
            return -2;
        }

        historial[i - 1].iter = i;
        historial[i - 1].x_n = xn;
        historial[i - 1].x_next = x_next;
        historial[i - 1].error = error;
        historial[i - 1].fx_next = fx;

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

static void print_biseccion_json(
    const Sistema *sistema,
    const IterBiseccion *h,
    int n,
    double raiz,
    int estado
) {
    printf("{\"ok\":true,\"metodo\":\"biseccion\",\"sistema\":\"%s\",", sistema->nombre);
    printf("\"raiz\":%.17g,\"iteraciones\":%d,\"residuo\":%.17g,", raiz, n, fabs(sistema->f(raiz)));
    printf("\"convergio\":%s,", estado == 0 ? "true" : "false");
    printf("\"historial\":[");
    for (int i = 0; i < n; i++) {
        if (i > 0) printf(",");
        printf(
            "{\"iter\":%d,\"a\":%.17g,\"b\":%.17g,\"c\":%.17g,\"fc\":%.17g,\"error\":%.17g}",
            h[i].iter,
            h[i].a,
            h[i].b,
            h[i].c,
            h[i].fc,
            h[i].error
        );
    }
    printf("]}\n");
}

static void print_punto_fijo_json(
    const Sistema *sistema,
    const IterPuntoFijo *h,
    int n,
    double raiz,
    int estado
) {
    printf("{\"ok\":true,\"metodo\":\"punto_fijo\",\"sistema\":\"%s\",", sistema->nombre);
    printf("\"raiz\":%.17g,\"iteraciones\":%d,\"residuo\":%.17g,", raiz, n, fabs(sistema->f(raiz)));
    printf("\"error_fijo\":%.17g,\"convergio\":%s,", fabs(sistema->g(raiz) - raiz), estado == 0 ? "true" : "false");
    printf("\"historial\":[");
    for (int i = 0; i < n; i++) {
        if (i > 0) printf(",");
        printf(
            "{\"iter\":%d,\"x_n\":%.17g,\"x_next\":%.17g,\"error\":%.17g,\"fx_next\":%.17g}",
            h[i].iter,
            h[i].x_n,
            h[i].x_next,
            h[i].error,
            h[i].fx_next
        );
    }
    printf("]}\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_error_json("Uso: algoritmos_api <biseccion|punto_fijo> ...");
        return 1;
    }

    const char *metodo = argv[1];

    if (strcmp(metodo, "biseccion") == 0) {
        if (argc != 7) {
            print_error_json("Parametros: biseccion <sistema_id> <a> <b> <tol> <max_iter>");
            return 1;
        }

        int sistema_id = 0;
        if (!parse_int(argv[2], &sistema_id) || sistema_id < 1 || sistema_id > NUM_SISTEMAS) {
            print_error_json("sistema_id invalido");
            return 1;
        }

        double a, b, tol;
        int max_iter;
        if (!parse_double(argv[3], &a) || !parse_double(argv[4], &b) || !parse_double(argv[5], &tol) || !parse_int(argv[6], &max_iter)) {
            print_error_json("Parametros numericos invalidos");
            return 1;
        }
        if (tol <= 0.0) {
            print_error_json("tol debe ser > 0");
            return 1;
        }

        IterBiseccion *hist = (IterBiseccion *)malloc((size_t)max_iter * sizeof(IterBiseccion));
        if (hist == NULL) {
            print_error_json("Sin memoria");
            return 1;
        }

        int n = 0;
        double raiz = NAN;
        int estado = biseccion(SISTEMAS[sistema_id - 1].f, a, b, tol, max_iter, hist, &n, &raiz);
        if (estado == -1) {
            free(hist);
            print_error_json("f(a) y f(b) tienen el mismo signo");
            return 1;
        }
        if (estado == -2) {
            free(hist);
            print_error_json("Aparecio NaN/Inf durante el calculo");
            return 1;
        }

        print_biseccion_json(&SISTEMAS[sistema_id - 1], hist, n, raiz, estado);
        free(hist);
        return 0;
    }

    if (strcmp(metodo, "punto_fijo") == 0) {
        if (argc != 6) {
            print_error_json("Parametros: punto_fijo <sistema_id> <x0> <tol> <max_iter>");
            return 1;
        }

        int sistema_id = 0;
        if (!parse_int(argv[2], &sistema_id) || sistema_id < 1 || sistema_id > NUM_SISTEMAS) {
            print_error_json("sistema_id invalido");
            return 1;
        }

        double x0, tol;
        int max_iter;
        if (!parse_double(argv[3], &x0) || !parse_double(argv[4], &tol) || !parse_int(argv[5], &max_iter)) {
            print_error_json("Parametros numericos invalidos");
            return 1;
        }
        if (tol <= 0.0) {
            print_error_json("tol debe ser > 0");
            return 1;
        }

        IterPuntoFijo *hist = (IterPuntoFijo *)malloc((size_t)max_iter * sizeof(IterPuntoFijo));
        if (hist == NULL) {
            print_error_json("Sin memoria");
            return 1;
        }

        int n = 0;
        double raiz = NAN;
        int estado = punto_fijo(
            SISTEMAS[sistema_id - 1].g,
            SISTEMAS[sistema_id - 1].f,
            x0,
            tol,
            max_iter,
            hist,
            &n,
            &raiz
        );
        if (estado == -2) {
            free(hist);
            print_error_json("Aparecio NaN/Inf durante el calculo");
            return 1;
        }

        print_punto_fijo_json(&SISTEMAS[sistema_id - 1], hist, n, raiz, estado);
        free(hist);
        return 0;
    }

    print_error_json("Metodo no soportado");
    return 1;
}
