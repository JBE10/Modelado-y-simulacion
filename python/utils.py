from typing import Any

def _res(raiz: float, iteraciones: int, historial: list[dict[str, Any]], convergio: bool, justificacion: str) -> dict[str, Any]:
    """Empaqueta el resultado estandarizado de los métodos iterativos."""
    return {
        "raiz": raiz,
        "iteraciones": iteraciones,
        "historial": historial,
        "convergio": convergio,
        "justificacion": justificacion
    }
