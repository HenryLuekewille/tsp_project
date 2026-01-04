import numpy as np
import random
import time
from typing import List, Tuple

class SimulatedAnnealingTSP:
    def __init__(self, cities: List[str], distance_matrix: np.ndarray,
                 start_city: str, temp_initial: float = 10000,
                 temp_min: float = 1, cooling_rate: float = 0.995):
        """
        Simulated Annealing für TSP
        
        Args:
            cities: Liste der Städtenamen
            distance_matrix: NxN Matrix mit Distanzen in km
            start_city: Startstadt
            temp_initial: Anfangstemperatur
            temp_min: Minimale Temperatur
            cooling_rate: Abkühlungsrate (0.9-0.999)
        """
        if start_city not in cities:
            raise ValueError(f"Startstadt '{start_city}' nicht in Liste")
        if len(cities) != distance_matrix.shape[0]:
            raise ValueError("Städte-Anzahl != Distanzmatrix-Dimension")
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("Distanzmatrix muss quadratisch sein")
        if cooling_rate <= 0 or cooling_rate >= 1:
            raise ValueError("Cooling Rate muss zwischen 0 und 1 liegen")
        
        self.cities = cities
        self.distance_matrix = distance_matrix
        self.start_idx = cities.index(start_city)
        self.temp_initial = temp_initial
        self.temp_min = temp_min
        self.cooling_rate = cooling_rate
        
        self.best_distance_history = []
        self.temperature_history = []
    
    def calculate_route_distance(self, route: List[int]) -> float:
        """Gesamtdistanz einer Route berechnen (inkl. Rückkehr)"""
        distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]  # Zurück zum Start
            distance += self.distance_matrix[from_city, to_city]
        return distance
    
    def get_neighbor_2opt(self, route: List[int]) -> List[int]:
        """
        Nachbar-Lösung durch 2-opt swap generieren
        Vertauscht zwei zufällige Städte (außer Startstadt)
        """
        new_route = route.copy()
        
        # Wähle zwei verschiedene Positionen (nicht die Startstadt)
        i, j = sorted(random.sample(range(1, len(route)), 2))
        
        # Reverse der Teilroute zwischen i und j
        new_route[i:j+1] = reversed(new_route[i:j+1])
        
        return new_route
    
    def acceptance_probability(self, old_cost: float, new_cost: float,
                               temperature: float) -> float:
        """Akzeptanzwahrscheinlichkeit nach Metropolis-Kriterium"""
        if new_cost < old_cost:
            return 1.0
        return np.exp((old_cost - new_cost) / temperature)
    
    def solve(self, verbose: bool = True) -> Tuple[List[str], float, dict]:
        """
        TSP mit Simulated Annealing lösen
        
        Returns:
            (route_names, distance, statistics)
        """
        start_time = time.time()
        
        # Initiale Lösung: Startstadt fest, Rest zufällig
        current_route = [self.start_idx]
        remaining = [i for i in range(len(self.cities)) 
                    if i != self.start_idx]
        random.shuffle(remaining)
        current_route.extend(remaining)
        
        current_cost = self.calculate_route_distance(current_route)
        best_route = current_route.copy()
        best_cost = current_cost
        
        temperature = self.temp_initial
        iterations = 0
        accepted = 0
        improvements = 0
        
        if verbose:
            print(f"\n🔥 Simulated Annealing")
            print(f"  Initiale Distanz: {current_cost:.2f} km")
            print(f"  Temperatur: {temperature:.0f} → {self.temp_min}")
        
        # Hauptschleife
        while temperature > self.temp_min:
            iterations += 1
            
            # Nachbar-Lösung generieren
            new_route = self.get_neighbor_2opt(current_route)
            new_cost = self.calculate_route_distance(new_route)
            
            # Akzeptanz-Entscheidung
            if random.random() < self.acceptance_probability(
                current_cost, new_cost, temperature
            ):
                current_route = new_route
                current_cost = new_cost
                accepted += 1
                
                # Beste Lösung aktualisieren
                if new_cost < best_cost:
                    best_route = new_route.copy()
                    best_cost = new_cost
                    improvements += 1
            
            # Tracking
            if iterations % 1000 == 0:
                self.best_distance_history.append(best_cost)
                self.temperature_history.append(temperature)
            
            # Temperatur reduzieren
            temperature *= self.cooling_rate
        
        runtime = time.time() - start_time
        
        # Route in Städtenamen umwandeln
        best_route_names = [self.cities[i] for i in best_route]
        
        # Statistiken
        stats = {
            'iterations': iterations,
            'runtime': runtime,
            'accepted_moves': accepted,
            'improvements': improvements,
            'acceptance_rate': accepted / iterations,
            'initial_distance': self.calculate_route_distance([self.start_idx] + remaining),
            'final_distance': best_cost
        }
        
        if verbose:
            print(f"  Finale Distanz: {best_cost:.2f} km")
            print(f"  Verbesserung: {stats['initial_distance'] - best_cost:.2f} km "
                  f"({((stats['initial_distance'] - best_cost) / stats['initial_distance'] * 100):.1f}%)")
            print(f"  Iterationen: {iterations:,}")
            print(f"  Laufzeit: {runtime:.3f}s")
            print(f"  Akzeptanzrate: {stats['acceptance_rate']:.1%}")
        
        return best_route_names, best_cost, stats