from __future__ import annotations
from sympy import lambdify

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

try:
    from sympy.calculus.util import function_range  # tipo: ignore
except Exception:  # pragma: no cover
    function_range = None  # type: ignore

# ================================================================
# Core
# ================================================================

def analizar_funcion(expr_str: str, var: str = "x") -> Tuple[object, Symbol]:
    """Convierte un string en expresión simbólica de SymPy y retorna (expr, símbolo)."""
    if not isinstance(expr_str, str) or not expr_str.strip():
        raise ValueError("expr_str debe ser un string no vacío")

    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    x = Symbol(var, real=True)
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
    expr = expr.subs(Symbol(var), x)
    return expr, x

def calcular_dominio(expr, var: str = "x") -> Set:
    """Calcula el dominio real (donde la función es continua) usando SymPy."""
    x = Symbol(var, real=True)
    try:
        dom = continuous_domain(expr, x, S.Reals)
        return dom
    except Exception:
        try:
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
    metodo: str  # "simbolico"
    detalle: Optional[str] = None

def calcular_recorrido(expr, var: str = "x",
                        dominio: Optional[Set] = None) -> RangoResultado:
    """Estima el rango (recorrido) de f: ℝ→ℝ usando solo el método simbólico."""
    x = Symbol(var, real=True)
    if dominio is None:
        dominio = calcular_dominio(expr, var)
    if function_range is not None:
        try:
            fr = function_range(expr, x, dominio)
            if fr is not EmptySet:
                return RangoResultado(fr, metodo="simbolico")
        except Exception:
            pass
    return RangoResultado(EmptySet, metodo="simbolico", detalle="No se pudo calcular el rango simbólicamente")

def calcular_intersecciones(expr, var: str = "x") -> Dict[str, object]:
    """Obtiene intersecciones con los ejes."""
    x = Symbol(var, real=True)
    try:
        x_inter = solveset(expr, x, domain=S.Reals)
    except Exception:
        try:
            x_inter = solveset_real(expr, x)
        except Exception:
            x_inter = EmptySet
    y_inter = None
    try:
        val0 = expr.subs(x, 0)
        if val0.is_real:
            y_inter = val0
    except Exception:
        y_inter = None
    return {"x": x_inter, "y": y_inter}

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