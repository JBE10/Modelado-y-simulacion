#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

typedef struct {
    const char *src;
    const char *p;
    double x;
    bool ok;
    char err[160];
} Parser;

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
    double residual;
} IterPuntoFijo;

static void set_error(Parser *ps, const char *msg) {
    if (ps->ok) {
        ps->ok = false;
        snprintf(ps->err, sizeof(ps->err), "%s", msg);
    }
}

static void skip_ws(Parser *ps) {
    while (*ps->p != '\0' && isspace((unsigned char)*ps->p)) {
        ps->p++;
    }
}

static bool match_char(Parser *ps, char c) {
    skip_ws(ps);
    if (*ps->p == c) {
        ps->p++;
        return true;
    }
    return false;
}

static double parse_expression(Parser *ps);

static double apply_func(const char *name, double v, bool *ok) {
    if (strcmp(name, "sin") == 0) return sin(v);
    if (strcmp(name, "cos") == 0) return cos(v);
    if (strcmp(name, "tan") == 0) return tan(v);
    if (strcmp(name, "exp") == 0) return exp(v);
    if (strcmp(name, "log") == 0 || strcmp(name, "ln") == 0) return log(v);
    if (strcmp(name, "sqrt") == 0) return sqrt(v);
    if (strcmp(name, "abs") == 0) return fabs(v);
    if (strcmp(name, "cbrt") == 0) return cbrt(v);
    *ok = false;
    return NAN;
}

static double parse_primary(Parser *ps) {
    skip_ws(ps);
    if (!ps->ok) {
        return NAN;
    }

    if (match_char(ps, '(')) {
        double v = parse_expression(ps);
        if (!match_char(ps, ')')) {
            set_error(ps, "Falta ')' en expresion");
        }
        return v;
    }

    if (*ps->p == '\0') {
        set_error(ps, "Expresion incompleta");
        return NAN;
    }

    if (isdigit((unsigned char)*ps->p) || *ps->p == '.') {
        char *end = NULL;
        double v = strtod(ps->p, &end);
        if (end == ps->p) {
            set_error(ps, "Numero invalido");
            return NAN;
        }
        ps->p = end;
        return v;
    }

    if (isalpha((unsigned char)*ps->p) || *ps->p == '_') {
        char id[64];
        int k = 0;
        while ((isalpha((unsigned char)*ps->p) || isdigit((unsigned char)*ps->p) || *ps->p == '_') && k < (int)sizeof(id) - 1) {
            id[k++] = *ps->p;
            ps->p++;
        }
        id[k] = '\0';

        if (strcmp(id, "x") == 0) {
            return ps->x;
        }
        if (strcmp(id, "pi") == 0) {
            return M_PI;
        }
        if (strcmp(id, "e") == 0) {
            return M_E;
        }

        if (!match_char(ps, '(')) {
            set_error(ps, "Se esperaba '(' despues de funcion");
            return NAN;
        }
        double arg = parse_expression(ps);
        if (!match_char(ps, ')')) {
            set_error(ps, "Falta ')' en llamada de funcion");
            return NAN;
        }
        bool ok = true;
        double out = apply_func(id, arg, &ok);
        if (!ok) {
            set_error(ps, "Funcion no soportada");
            return NAN;
        }
        return out;
    }

    set_error(ps, "Token no valido en expresion");
    return NAN;
}

static double parse_unary(Parser *ps) {
    skip_ws(ps);
    if (match_char(ps, '+')) return parse_unary(ps);
    if (match_char(ps, '-')) return -parse_unary(ps);
    return parse_primary(ps);
}

static bool match_pow(Parser *ps) {
    skip_ws(ps);
    if (*ps->p == '^') {
        ps->p++;
        return true;
    }
    if (ps->p[0] == '*' && ps->p[1] == '*') {
        ps->p += 2;
        return true;
    }
    return false;
}

static double parse_power(Parser *ps) {
    double left = parse_unary(ps);
    if (!ps->ok) return NAN;
    if (match_pow(ps)) {
        double right = parse_power(ps);
        return pow(left, right);
    }
    return left;
}

static double parse_term(Parser *ps) {
    double v = parse_power(ps);
    while (ps->ok) {
        skip_ws(ps);
        if (*ps->p == '*' && ps->p[1] != '*') {
            ps->p++;
            v *= parse_power(ps);
        } else if (*ps->p == '/') {
            ps->p++;
            double d = parse_power(ps);
            v /= d;
        } else {
            break;
        }
    }
    return v;
}

static double parse_expression(Parser *ps) {
    double v = parse_term(ps);
    while (ps->ok) {
        skip_ws(ps);
        if (*ps->p == '+') {
            ps->p++;
            v += parse_term(ps);
        } else if (*ps->p == '-') {
            ps->p++;
            v -= parse_term(ps);
        } else {
            break;
        }
    }
    return v;
}

static bool eval_expr(const char *expr, double x, double *out, char *err, size_t err_sz) {
    Parser ps;
    ps.src = expr;
    ps.p = expr;
    ps.x = x;
    ps.ok = true;
    ps.err[0] = '\0';

    double v = parse_expression(&ps);
    skip_ws(&ps);
    if (ps.ok && *ps.p != '\0') {
        set_error(&ps, "Sobran caracteres al final");
    }
    if (ps.ok && !isfinite(v)) {
        set_error(&ps, "Resultado no finito (NaN/Inf)");
    }

    if (!ps.ok) {
        snprintf(err, err_sz, "%s", ps.err[0] ? ps.err : "Error de parseo");
        return false;
    }
    *out = v;
    return true;
}

static void print_error_json(const char *msg) {
    printf("{\"ok\":false,\"error\":\"%s\"}\n", msg);
}

static bool parse_double(const char *s, double *out) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (s == end || *end != '\0' || !isfinite(v)) return false;
    *out = v;
    return true;
}

static bool parse_int(const char *s, int *out) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s == end || *end != '\0' || v < 1 || v > 1000000) return false;
    *out = (int)v;
    return true;
}

static bool eval_f_expr(const char *expr, double x, double *out, char *err, size_t err_sz) {
    return eval_expr(expr, x, out, err, err_sz);
}

static bool estimar_derivada(const char *expr_g, double x, double *gprime, char *err, size_t err_sz) {
    double h = 1e-6 * (1.0 + fabs(x));
    double gp, gm;
    if (!eval_expr(expr_g, x + h, &gp, err, err_sz)) return false;
    if (!eval_expr(expr_g, x - h, &gm, err, err_sz)) return false;
    *gprime = (gp - gm) / (2.0 * h);
    if (!isfinite(*gprime)) {
        snprintf(err, err_sz, "No se pudo estimar g'(x) de forma numerica");
        return false;
    }
    return true;
}

static int biseccion(
    const char *expr_f,
    double a,
    double b,
    double tol,
    int max_iter,
    IterBiseccion *historial,
    int *n_historial,
    double *raiz,
    char *justif,
    size_t justif_sz,
    char *err,
    size_t err_sz
) {
    double a0 = a;
    double b0 = b;
    double fa, fb;
    if (!eval_f_expr(expr_f, a, &fa, err, err_sz)) {
        snprintf(err, err_sz, "No se pudo evaluar f(a) en a=%.10g", a);
        return -3;
    }
    if (!eval_f_expr(expr_f, b, &fb, err, err_sz)) {
        snprintf(err, err_sz, "No se pudo evaluar f(b) en b=%.10g", b);
        return -3;
    }
    if (fa * fb > 0.0) return -1;

    for (int i = 1; i <= max_iter; i++) {
        double c = 0.5 * (a + b);
        double fc;
        if (!eval_f_expr(expr_f, c, &fc, err, err_sz)) {
            int n_prev = i - 1;
            *n_historial = n_prev;
            *raiz = 0.5 * (a + b);
            snprintf(
                justif,
                justif_sz,
                "Se detuvo por error de evaluacion en iter %d: no se pudo calcular f(c) con c=%.10g.",
                i,
                c
            );
            return 2;
        }
        double error = 0.5 * fabs(b - a);

        historial[i - 1].iter = i;
        historial[i - 1].a = a;
        historial[i - 1].b = b;
        historial[i - 1].c = c;
        historial[i - 1].fc = fc;
        historial[i - 1].error = error;

        if (fabs(fc) < tol || error < tol) {
            *n_historial = i;
            *raiz = c;
            snprintf(justif, justif_sz, "Convergio: se cumplio |f(c)| < tol o error < tol en la iteracion %d.", i);
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
    {
        double ancho0 = fabs(b0 - a0);
        double requerido = (ancho0 > 0.0 && tol > 0.0) ? ceil(log2(ancho0 / tol)) : (double)max_iter;
        snprintf(
            justif,
            justif_sz,
            "No convergio: se alcanzo max_iter=%d. Para este intervalo y tolerancia, se estiman %.0f iteraciones de biseccion.",
            max_iter,
            requerido
        );
    }
    return 1;
}

static int punto_fijo(
    const char *expr_g,
    const char *expr_f,
    double x0,
    double tol,
    int max_iter,
    IterPuntoFijo *historial,
    int *n_historial,
    double *raiz,
    double *residuo_final,
    char *justif,
    size_t justif_sz,
    char *err,
    size_t err_sz
) {
    double xn = x0;
    for (int i = 1; i <= max_iter; i++) {
        double x_next;
        if (!eval_expr(expr_g, xn, &x_next, err, err_sz)) {
            int n_prev = i - 1;
            *n_historial = n_prev;
            *raiz = xn;
            *residuo_final = (n_prev > 0) ? historial[n_prev - 1].residual : 0.0;
            snprintf(
                justif,
                justif_sz,
                "Se detuvo por error de evaluacion en iter %d: no se pudo calcular g(x_n) con x_n=%.10g.",
                i,
                xn
            );
            return 2;
        }
        double error = fabs(x_next - xn);
        double residual;
        if (!eval_expr(expr_f, x_next, &residual, err, err_sz)) {
            int n_prev = i - 1;
            *n_historial = n_prev;
            *raiz = xn;
            *residuo_final = (n_prev > 0) ? historial[n_prev - 1].residual : 0.0;
            snprintf(
                justif,
                justif_sz,
                "Se detuvo por error de evaluacion en iter %d: no se pudo calcular f(x_n+1) con x_n+1=%.10g.",
                i,
                x_next
            );
            return 2;
        }
        residual = fabs(residual);

        historial[i - 1].iter = i;
        historial[i - 1].x_n = xn;
        historial[i - 1].x_next = x_next;
        historial[i - 1].error = error;
        historial[i - 1].residual = residual;

        if (error < tol) {
            *n_historial = i;
            *raiz = x_next;
            *residuo_final = residual;
            snprintf(justif, justif_sz, "Convergio: se cumplio |x_n+1 - x_n| < tol en la iteracion %d.", i);
            return 0;
        }

        xn = x_next;
    }

    *n_historial = max_iter;
    *raiz = xn;
    double fr;
    if (!eval_expr(expr_f, *raiz, &fr, err, err_sz)) return -3;
    *residuo_final = fabs(fr);
    {
        double gprime;
        char derr[160] = {0};
        if (estimar_derivada(expr_g, *raiz, &gprime, derr, sizeof(derr))) {
            if (fabs(gprime) >= 1.0) {
                snprintf(
                    justif,
                    justif_sz,
                    "No convergio: se alcanzo max_iter=%d y |g'(x)|≈%.4f >= 1 cerca de la ultima aproximacion.",
                    max_iter,
                    fabs(gprime)
                );
            } else {
                snprintf(
                    justif,
                    justif_sz,
                    "No convergio: se alcanzo max_iter=%d. Se estimo |g'(x)|≈%.4f (<1), probablemente faltaron iteraciones o un x0 mas cercano.",
                    max_iter,
                    fabs(gprime)
                );
            }
        } else {
            snprintf(
                justif,
                justif_sz,
                "No convergio: se alcanzo max_iter=%d y no se pudo estimar g'(x) cerca de la ultima aproximacion.",
                max_iter
            );
        }
    }
    return 1;
}

static void print_biseccion_json(const IterBiseccion *h, int n, double raiz, double residuo, const char *justif, int estado) {
    printf("{\"ok\":true,\"metodo\":\"biseccion\",");
    printf("\"raiz\":%.17g,\"iteraciones\":%d,\"residuo\":%.17g,", raiz, n, residuo);
    printf("\"convergio\":%s,", estado == 0 ? "true" : "false");
    printf("\"terminado_por_error\":%s,", estado == 2 ? "true" : "false");
    printf("\"justificacion\":\"%s\",", justif);
    printf("\"historial\":[");
    for (int i = 0; i < n; i++) {
        if (i > 0) printf(",");
        printf(
            "{\"iter\":%d,\"a\":%.17g,\"b\":%.17g,\"c\":%.17g,\"fc\":%.17g,\"error\":%.17g}",
            h[i].iter, h[i].a, h[i].b, h[i].c, h[i].fc, h[i].error
        );
    }
    printf("]}\n");
}

static void print_punto_fijo_json(const IterPuntoFijo *h, int n, double raiz, double residuo, const char *justif, int estado) {
    printf("{\"ok\":true,\"metodo\":\"punto_fijo\",");
    printf("\"raiz\":%.17g,\"iteraciones\":%d,\"residuo\":%.17g,", raiz, n, residuo);
    printf("\"convergio\":%s,", estado == 0 ? "true" : "false");
    printf("\"terminado_por_error\":%s,", estado == 2 ? "true" : "false");
    printf("\"justificacion\":\"%s\",", justif);
    printf("\"historial\":[");
    for (int i = 0; i < n; i++) {
        if (i > 0) printf(",");
        printf(
            "{\"iter\":%d,\"x_n\":%.17g,\"x_next\":%.17g,\"error\":%.17g,\"residual\":%.17g}",
            h[i].iter, h[i].x_n, h[i].x_next, h[i].error, h[i].residual
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
            print_error_json("Parametros: biseccion <expr_f> <a> <b> <tol> <max_iter>");
            return 1;
        }
        const char *expr_f = argv[2];
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

        char err[160] = {0};
        char justif[320] = {0};
        int n = 0;
        double raiz = NAN;
        int estado = biseccion(expr_f, a, b, tol, max_iter, hist, &n, &raiz, justif, sizeof(justif), err, sizeof(err));
        if (estado == -1) {
            free(hist);
            print_error_json("f(a) y f(b) tienen el mismo signo");
            return 1;
        }
        if (estado == -3) {
            free(hist);
            print_error_json(err[0] ? err : "Error evaluando f(x)");
            return 1;
        }
        double residuo;
        if (estado != 2 && !eval_expr(expr_f, raiz, &residuo, err, sizeof(err))) {
            free(hist);
            print_error_json(err[0] ? err : "Error evaluando residuo");
            return 1;
        }
        if (estado == 2) {
            residuo = (n > 0) ? fabs(hist[n - 1].fc) : 0.0;
        }
        print_biseccion_json(hist, n, raiz, fabs(residuo), justif, estado);
        free(hist);
        return 0;
    }

    if (strcmp(metodo, "punto_fijo") == 0) {
        if (argc != 7) {
            print_error_json("Parametros: punto_fijo <expr_g> <x0> <tol> <max_iter> <expr_f>");
            return 1;
        }
        const char *expr_g = argv[2];
        const char *expr_f = argv[6];
        if (expr_f[0] == '\0') {
            print_error_json("Para punto_fijo debes ingresar f(x) y g(x)");
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

        char err[160] = {0};
        char justif[320] = {0};
        int n = 0;
        double raiz = NAN;
        double residuo = NAN;
        int estado = punto_fijo(expr_g, expr_f, x0, tol, max_iter, hist, &n, &raiz, &residuo, justif, sizeof(justif), err, sizeof(err));
        if (estado == -3) {
            free(hist);
            print_error_json(err[0] ? err : "Error evaluando expresiones");
            return 1;
        }

        print_punto_fijo_json(hist, n, raiz, residuo, justif, estado);
        free(hist);
        return 0;
    }

    print_error_json("Metodo no soportado");
    return 1;
}
