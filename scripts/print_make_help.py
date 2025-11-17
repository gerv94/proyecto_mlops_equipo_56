#!/usr/bin/env python
"""
Script para parsear el Makefile y mostrar ayuda formateada.
Extrae los comentarios ## de cada target y los muestra en formato legible.
"""

import re
import sys
from pathlib import Path


def parse_makefile(makefile_path: str) -> dict:
    """Parsea el Makefile y extrae targets con sus descripciones."""
    targets = {}
    
    try:
        with open(makefile_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Makefile no encontrado: {makefile_path}", file=sys.stderr)
        return targets
    
    # Buscar patrones: target: ## descripción
    # También captura targets con dependencias: target: dep1 dep2 ## descripción
    pattern = r'^([a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)?)\s*:.*?##\s*(.+)$'
    
    for line in content.split('\n'):
        match = re.match(pattern, line)
        if match:
            target = match.group(1)
            description = match.group(2).strip()
            targets[target] = description
    
    return targets


def print_help(targets: dict):
    """Imprime la ayuda formateada."""
    if not targets:
        print("No se encontraron comandos documentados en el Makefile.")
        return
    
    # Ordenar alfabéticamente
    sorted_targets = sorted(targets.items())
    
    # Encontrar el ancho máximo del nombre del target para alineación
    max_width = max(len(target) for target in targets.keys()) if targets else 0
    
    print("\nComandos disponibles:\n")
    for target, description in sorted_targets:
        # Saltar .PHONY y otros targets especiales
        if target.startswith('.'):
            continue
        print(f"  make {target:<{max_width}}  {description}")


def main():
    """Función principal."""
    # Obtener archivos Makefile de los argumentos
    makefiles = sys.argv[1:] if len(sys.argv) > 1 else ['Makefile']
    
    all_targets = {}
    for makefile in makefiles:
        targets = parse_makefile(makefile)
        all_targets.update(targets)
    
    print_help(all_targets)


if __name__ == '__main__':
    main()

