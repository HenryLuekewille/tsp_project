import numpy as np
import time
from itertools import permutations
from typing import List, Tuple, Dict

def exact_tsp_brute_force(cities: List[str], distance_matrix: np.ndarray,
                          start_city: str, max_cities: int = 12,
                          verbose: bool = True) -> Tuple[List[str], float, Dict]:
    """
    Exakte TSP-Lösung durch vollständige Enumeration (Brute Force)
    
    Args:
        cities: Liste der Städtenamen
        distance_matrix: NxN Distanzmatrix
        start_city: Startstadt
        max_cities: Sicherheitsgrenze (n! wächst sehr schnell!)
        verbose: Output anzeigen
    
    Returns:
        (route, distance, statistics)
    
    Komplexität: O(n!)
    """
    if start_city not in cities:
        raise ValueError(f"Startstadt '{start_city}' nicht in Liste")
    
    n = len(cities)
    
    if n > max_cities:
        raise ValueError(
            f"Zu viele Städte für Brute Force ({n} > {max_cities}).\n"
            f"Fakultät {n}! = {np.math.factorial(n):,} Permutationen!\n"
            f"Geschätzte Laufzeit: {np.math.factorial(n) / 1e6:.1f}s bei 1M perm/s"
        )
    
    start_time = time.time()
    start_idx = cities.index(start_city)
    other_cities = [i for i in range(n) if i != start_idx]
    
    min_distance = float('inf')
    best_route = None
    total_permutations = np.math.factorial(len(other_cities))
    
    if verbose:
        print(f"\n🎯 Exakte Lösung (Brute Force)")
        print(f"  Städte: {n}")
        print(f"  Permutationen: {total_permutations:,}")
    
    # Alle Permutationen durchprobieren
    for perm in permutations(other_cities):
        route = [start_idx] + list(perm)
        
        # Distanz berechnen (inkl. Rückkehr zum Start)
        distance = sum(
            distance_matrix[route[i], route[(i+1) % len(route)]]
            for i in range(len(route))
        )
        
        if distance < min_distance:
            min_distance = distance
            best_route = route
    
    runtime = time.time() - start_time
    
    # Route in Städtenamen umwandeln
    best_route_names = [cities[i] for i in best_route]
    
    stats = {
        'iterations': total_permutations,
        'runtime': runtime,
        'permutations_per_second': total_permutations / runtime
    }
    
    if verbose:
        print(f"  Optimale Distanz: {min_distance:.2f} km")
        print(f"  Laufzeit: {runtime:.3f}s")
        print(f"  Geschwindigkeit: {stats['permutations_per_second']:,.0f} perm/s")
    
    return best_route_names, min_distance, stats