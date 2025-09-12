"""
Notas:
- Se prioriza trabajar sobre los reales: ℝ.
- `calcular_recorrido` intenta primero método simbólico; si no aplica,
  usa un muestreo numérico robusto (fallback) para estimar el rango.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable
from sympy import (
    S, Symbol, sympify, Interval, Union, EmptySet, FiniteSet, oo,
    Abs, exp, log, sqrt, re, im
)
from sympy.calculus.util import continuous_domain
from sympy.sets.sets import Set
from sympy.solvers.solveset import solveset, solveset_real
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)
import numpy as np
from sympy import lambdify

# Algunas funciones pueden no estar presentes en versiones muy antiguas de SymPy
try:
    from sympy.calculus.util import function_range  # tipo: ignore
except Exception:  # pragma: no cover
    function_range = None  # type: ignore

# ================================================================
# Core
# ================================================================

def analizar_funcion(expr_str: str, var: str = "x") -> Tuple[object, Symbol]:
    """Convierte un string en expresión simbólica de SymPy y retorna (expr, símbolo).

    Soporta multiplicación implícita ("2x", "x(x+1)") y ^ como potencia.

    Ejemplos:
        >>> expr, x = analizar_funcion("(x^2-9)/(x-3)")
        >>> expr
        (x**2 - 9)/(x - 3)
    """
    if not isinstance(expr_str, str) or not expr_str.strip():
        raise ValueError("expr_str debe ser un string no vacío")

    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    x = Symbol(var, real=True)
    # Permitir funciones comunes por seguridad (sympify/parse_expr es estricto)
    local_dict = {
        var: x,
        'abs': Abs,
        'sqrt': sqrt,
        'log': log,
        'exp': exp,
        'e': S.Exp1,
        'pi': S.Pi,
    }
    expr = parse_expr(expr_str, transformations=transformations,
                      local_dict=local_dict, evaluate=True)
    # Fuerza reemplazo del símbolo por el Symbol(real=True) creado arriba
    expr = expr.subs(Symbol(var), x)
    return expr, x


def calcular_dominio(expr, var: str = "x") -> Set:
    """Calcula el dominio real (donde la función es continua) usando SymPy.

    Para funciones con discontinuidades removibles, devuelve el conjunto
    donde es continua. Si se requiere el *dominio de definición* (incluye
    puntos aislados donde existe el valor), considerar añadir esos puntos
    manualmente evaluando límites.
    """
    x = Symbol(var, real=True)
    try:
        dom = continuous_domain(expr, x, S.Reals)
        return dom
    except Exception:
        # Fallback conservador: intentar excluir singularidades evidentes
        try:
            # Excluir ceros del denominador si es fracción racional
            den = expr.as_numer_denom()[1]
            den_zeros = solveset_real(den, x)
            if den_zeros is S.Reals:
                return S.Reals
            if isinstance(den_zeros, (FiniteSet, Interval, Union)):
                return S.Reals - den_zeros
        except Exception:
            pass
        return S.Reals  # último recurso


@dataclass
class RangoResultado:
    conjunto: Set
    metodo: str  # "simbolico" | "muestreo"
    detalle: Optional[str] = None


def calcular_recorrido(expr, var: str = "x",
                        dominio: Optional[Set] = None,
                        muestreo_intervalos: Optional[Iterable[Interval]] = None,
                        puntos_por_intervalo: int = 400,
                        umbral_infinito: float = 1e6) -> RangoResultado:
    """Estima el rango (recorrido) de f: ℝ→ℝ.

    Estrategia:
      1) Intento simbólico con `function_range` en el dominio dado (o continuo).
      2) Si falla o retorna vació, estimación por muestreo numérico robusto:
         - Malla densa en intervalos del dominio (por defecto [-10,10] y
           alrededores de posibles asíntotas si se detectan).

    Parámetros clave:
      - `puntos_por_intervalo`: densidad de muestreo (↑ = más preciso, más costo).
      - `umbral_infinito`: si |f(x)| excede este valor en muestreo, se asume
        tendencia a ±∞ y se abre el intervalo correspondiente.
    """
    x = Symbol(var, real=True)

    # 1) Dominio por defecto = dominio de continuidad en Reals
    if dominio is None:
        dominio = calcular_dominio(expr, var)

    # 2) Intento simbólico
    if function_range is not None:
        try:
            fr = function_range(expr, x, dominio)
            if fr is not EmptySet:
                return RangoResultado(fr, metodo="simbolico")
        except Exception:
            pass

    # 3) Fallback por muestreo
    try:
        if muestreo_intervalos is None:
            # Construye intervalos razonables según el dominio
            muestreo_intervalos = []
            # Base: [-10, 10]
            muestreo_intervalos.append(Interval(-10, 10))
            # Si el dominio es todo Reals, añade extremos más lejanos
            if dominio == S.Reals:
                muestreo_intervalos.extend([Interval(-100, -10), Interval(10, 100)])

        f = lambdify(x, expr, modules=["numpy"])  # vectorizado

        valores = []
        infinito_pos = False
        infinito_neg = False

        for I in muestreo_intervalos:
            a, b = float(I.inf), float(I.sup)
            xs = np.linspace(a, b, puntos_por_intervalo, dtype=float)
            # Evitar evaluar exactamente en singularidades simples
            # Detectar posibles ceros del denominador para evitar divisiones por 0
            den = expr.as_numer_denom()[1]
            den_f = lambdify(x, den, modules=["numpy"]) if den != 1 else None
            if den_f is not None:
                mask = np.ones_like(xs, dtype=bool)
                try:
                    den_vals = den_f(xs)
                    mask &= np.abs(den_vals) > 1e-9
                except Exception:
                    pass
                xs = xs[mask]
            if xs.size == 0:
                continue
            try:
                ys = f(xs)
            except Exception:
                # En caso de error numérico puntual, intenta evaluar punto a punto
                ys = []
                for xi in xs:
                    try:
                        yi = f(float(xi))
                        ys.append(yi)
                    except Exception:
                        continue
                if not ys:
                    continue
            ys = np.array(ys, dtype=float)
            # Filtrar nans e infs
            ys = ys[np.isfinite(ys)]
            if ys.size == 0:
                continue
            if np.any(ys > umbral_infinito):
                infinito_pos = True
            if np.any(ys < -umbral_infinito):
                infinito_neg = True
            valores.append((float(np.min(ys)), float(np.max(ys))))

        if not valores:
            return RangoResultado(EmptySet, metodo="muestreo",
                                  detalle="No se pudo muestrear el dominio")

        vmin = min(v[0] for v in valores)
        vmax = max(v[1] for v in valores)

        # Construir intervalo con posibles infinitos
        left = -oo if infinito_neg else vmin
        right = oo if infinito_pos else vmax
        conjunto = Interval(left, right)
        return RangoResultado(conjunto, metodo="muestreo",
                              detalle=f"muestras={puntos_por_intervalo}")
    except Exception as e:
        return RangoResultado(EmptySet, metodo="muestreo",
                              detalle=f"Error muestreo: {e}")


def calcular_intersecciones(expr, var: str = "x") -> Dict[str, object]:
    """Obtiene intersecciones con los ejes.

    Retorna un diccionario:
      {
        "x": Set de raíces reales (x-intercepts),
        "y": valor en x=0 si está definido (y-intercept) o None
      }
    """
    x = Symbol(var, real=True)

    # Intersecciones con eje X: resolver f(x)=0 en ℝ
    try:
        x_inter = solveset(expr, x, domain=S.Reals)
    except Exception:
        try:
            x_inter = solveset_real(expr, x)
        except Exception:
            x_inter = EmptySet

    # Intersección con eje Y: f(0) si está definido
    y_inter = None
    try:
        val0 = expr.subs(x, 0)
        if val0.is_real:
            y_inter = val0
    except Exception:
        y_inter = None

    return {"x": x_inter, "y": y_inter}


# ================================================================
# Helpers de presentación/depuración (opcional)
# ================================================================

def _to_latex(obj) -> str:
    try:
        from sympy import latex
        return latex(obj)
    except Exception:
        return str(obj)


def describir_resultados(expr, var: str = "x") -> str:
    """Devuelve un string breve con dominio, rango (método) e intersecciones."""
    dom = calcular_dominio(expr, var)
    rango = calcular_recorrido(expr, var)
    inter = calcular_intersecciones(expr, var)
    return (
        f"f({var}) = {expr}\n"
        f"Dominio: {dom}\n"
        f"Recorrido: {rango.conjunto} (método: {rango.metodo})\n"
        f"Intersecciones: eje X: {inter['x']}, eje Y: {inter['y']}\n"
    )


if __name__ == "__main__":
    # Pequeña demo manual
    pruebas = [
        "x^2 - 4",
        "(x^2 - 9)/(x-3)",
        "sqrt(9 - x^2)",
        "log(x-1)",
        "(x-1)/(x^2-1)",
        "abs(x) + 1/x",
    ]
    for s in pruebas:
        expr, x = analizar_funcion(s)
        print("="*60)
        print(describir_resultados(expr))
