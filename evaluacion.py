
"""
evaluacion.py — Evaluación Paso a Paso (Matías)
Compatible con la salida de 'analizar_funcion' de Alan.
Enfocado en ℝ y con soporte para discontinuidades removibles (límites).

API principal
-------------
evaluar_punto(expr, x0, *, var="x", ndigits=8, dominio=None,
              usar_limite=True, simplificar_antes=True, solo_reales=True)

Devuelve EvaluacionResultado:
    - exacto:     valor simbólico exacto o None
    - decimal:    float o None
    - pasos:      [str]
    - latex:      dict (etiquetas -> LaTeX)
    - fuera_de_dominio: bool
    - error:      str|None
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from sympy import (
    symbols, Symbol, S, simplify, Eq, N, latex, zoo, nan, oo, E, pi,
    limit, sympify, Interval
)
from sympy.core.expr import Expr
from sympy.sets.sets import Set
from sympy.calculus.util import continuous_domain


@dataclass
class EvaluacionResultado:
    exacto: Optional[Expr]
    decimal: Optional[float]
    pasos: List[str]
    latex: Dict[str, str]
    fuera_de_dominio: bool
    error: Optional[str] = None


def _make_locals(var: str):
    return {"e": E, "pi": pi, var: symbols(var, real=True)}


def _is_bad(value: Expr) -> bool:
    try:
        if value.has(zoo) or value.has(nan) or value.has(oo) or value.has(-oo):
            return True
        if value.is_infinite is True:
            return True
    except Exception:
        return False
    return False


def _esta_en_dominio(x0, dominio: Optional[Set]) -> bool:
    if dominio is None:
        return True
    try:
        # Si el dominio es un conjunto, revisa pertenencia
        return (S(x0) in dominio) is True
    except Exception:
        return True


def evaluar_punto(expr: Expr,
                  x0,
                  *,
                  var: str = "x",
                  ndigits: int = 8,
                  dominio: Optional[Set] = None,
                  usar_limite: bool = True,
                  simplificar_antes: bool = True,
                  solo_reales: bool = True) -> EvaluacionResultado:
    """
    Evalúa f(x) en x0 mostrando pasos (versión ℝ).
    - Si 'dominio' no se entrega, se usa continuous_domain(expr, x, S.Reals).
    - Si sustitución da indeterminación pero existe límite finito, lo usa (si usar_limite=True).
    - 'solo_reales': si el resultado exacto no es real, lo reporta como fuera de dominio.
    """
    pasos: List[str] = []
    latex_parts: Dict[str, str] = {}

    try:
        x: Symbol = symbols(var, real=True)
        pasos.append(f"1) Expresión original (ℝ): f({var}) = {expr}")
        latex_parts["expr_original"] = latex(Eq(Symbol(f"f({var})"), expr))

        # Dominio por defecto (continuidad en ℝ), coherente con Alan
        if dominio is None:
            try:
                dominio = continuous_domain(expr, x, S.Reals)
                pasos.append(f"2) Dominio de continuidad estimado: {dominio}")
                latex_parts["dominio"] = latex(dominio)
            except Exception:
                dominio = None
                pasos.append("2) No se pudo determinar dominio; se asume ℝ (con validación puntual).")

        # Opción (útil p/ (x-1)/(x-1)): simplificar ANTES de sustituir
        expr_proc = simplify(expr) if simplificar_antes else expr
        if simplificar_antes and expr_proc is not expr:
            pasos.append(f"3) Simplificación previa: {expr}  →  {expr_proc}")
            latex_parts["simplif_previa"] = latex(expr_proc)
        else:
            pasos.append("3) Sin simplificación previa (opción desactivada o no cambió).")
            latex_parts["simplif_previa"] = latex(expr_proc)

        x0S = S(x0)
        pertenece = _esta_en_dominio(x0S, dominio)
        if dominio is not None:
            pasos.append(f"4) ¿x0 ∈ dominio? → {'Sí' if pertenece else 'No'}")
        else:
            pasos.append("4) Dominio no explícito; se validará por evaluación directa/limit.")

        # Sustitución directa
        expr_sub = expr_proc.subs(x, x0S)
        pasos.append(f"5) Sustitución: f({x0S}) = {expr_sub}")
        latex_parts["sustitucion"] = latex(Eq(Symbol(f"f({x0S})"), expr_sub))

        # Caso directo fuera de dominio (div/0, NaN, ±∞)
        if _is_bad(expr_sub):
            pasos.append("6) La sustitución directa indica indeterminación/∞/NaN.")
            # Intentar límite si está pedido y el punto no pertenece al dominio de continuidad
            if usar_limite:
                try:
                    lim = limit(expr_proc, x, x0S)
                    pasos.append(f"7) Límite al aproximar x→{x0S}: {lim}")
                    latex_parts["limite"] = latex(Eq(Symbol(f"\\lim_{{x\\to {latex(x0S)}}} f(x)"), lim))
                    if not _is_bad(lim) and (not solo_reales or (lim.is_real is True)):
                        exacto = lim
                        decimal = float(N(exacto, ndigits + 2)) if exacto.is_number else None
                        if decimal is not None:
                            pasos.append(f"8) Valor decimal (desde límite): {decimal:.{ndigits}g}")
                            latex_parts["decimal"] = latex(N(exacto, ndigits))
                        # Si estaba fuera del dominio formal pero el límite existe, reportamos como
                        # “valor por continuidad” (útil didácticamente; la GUI puede indicarlo).
                        pasos.append("9) Interpretación: discontinuidad removible → se usa el valor del límite.")
                        latex_parts["exacto"] = latex(exacto)
                        return EvaluacionResultado(exacto, decimal, pasos, latex_parts, fuera_de_dominio=False)
                    else:
                        pasos.append("7b) Límite no finito o no real → fuera de dominio.")
                        return EvaluacionResultado(None, None, pasos, latex_parts, True)
                except Exception as e:
                    pasos.append(f"7) No fue posible calcular el límite ({e}). → fuera de dominio.")
                    return EvaluacionResultado(None, None, pasos, latex_parts, True)
            # Si no usamos límite:
            return EvaluacionResultado(None, None, pasos, latex_parts, True)

        # Si la sustitución dio una expresión válida, intenta simplificar
        expr_simpl = simplify(expr_sub)
        if expr_simpl != expr_sub:
            pasos.append(f"6) Simplificación posterior: {expr_sub}  →  {expr_simpl}")
            latex_parts["simplif_posterior"] = latex(expr_simpl)
        else:
            pasos.append("6) La expresión ya está simplificada tras la sustitución.")
            latex_parts["simplif_posterior"] = latex(expr_simpl)

        exacto = expr_simpl
        # Si solo_reales, descartar resultados no reales
        if solo_reales and hasattr(exacto, "is_real") and exacto.is_real is False:
            pasos.append("7) El valor exacto no es real → fuera de dominio (modo solo_reales).")
            return EvaluacionResultado(None, None, pasos, latex_parts, True)

        pasos.append(f"7) Valor exacto: {exacto}")
        latex_parts["exacto"] = latex(exacto)

        decimal = None
        if exacto.is_number if hasattr(exacto, "is_number") else False:
            decimal = float(N(exacto, ndigits + 2))
            pasos.append(f"8) Valor decimal: {decimal:.{ndigits}g}")
            latex_parts["decimal"] = latex(N(exacto, ndigits))

        return EvaluacionResultado(exacto, decimal, pasos, latex_parts, False)

    except Exception as e:
        pasos.append("✗ Se produjo un error durante la evaluación.")
        return EvaluacionResultado(None, None, pasos, latex_parts, False, str(e))


def evaluar_punto_desde_str(expr_str: str, x0, *, var: str = "x", ndigits: int = 8, **kwargs) -> EvaluacionResultado:
    """Versión rápida para pruebas locales (en el proyecto usarán la Expr de Alan)."""
    try:
        locs = _make_locals(var)
        expr = sympify(expr_str, locals=locs)
    except Exception as e:
        return EvaluacionResultado(None, None, ["✗ No se pudo interpretar la expresión."], {}, False, f"sympify: {e}")
    return evaluar_punto(expr, x0, var=var, ndigits=ndigits, **kwargs)


if __name__ == "__main__":
    from sympy import sin, sqrt
    pruebas = [
        ("(x-1)/(x-1)", 1),
        ("sin(x)/x", 0),
        ("1/(x-1)", 1),
        ("sqrt(x-4)", 9),
        ("sqrt(x-4)", 1),  # fuera de dominio en ℝ
    ]
    for s, v in pruebas:
        print("="*70)
        print(f"f(x)={s}, x0={v}")
        r = evaluar_punto_desde_str(s, v)
        for p in r.pasos: print(p)
        if r.fuera_de_dominio: print("→ FUERA DE DOMINIO")
        elif r.error: print("→ ERROR:", r.error)
        else: print("→ Exacto:", r.exacto, "| Decimal:", r.decimal)
