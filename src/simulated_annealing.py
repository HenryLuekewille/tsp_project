"""
Simulated Annealing für das Travelling Salesman Problem (TSP)

Implementiert Simulated Annealing mit vollständigem Error Handling.
"""

import numpy as np
from typing import List, Tuple, Dict
import time
import random


class SimulatedAnnealingTSP:
    """
    Simulated Annealing zur Lösung des Travelling Salesman Problems.
    
    Der Algorithmus simuliert den physikalischen Abkühlungsprozess:
    - Hohe Temperatur: Akzeptiert auch schlechtere Lösungen (Exploration)
    - Niedrige Temperatur: Nur bessere Lösungen (Exploitation)
    
    Attributes:
        cities (List[str]): Liste der Städtenamen
        distance_matrix (np.ndarray): Symmetrische Distanzmatrix
        start_idx (int): Index der Startstadt
        temp_initial (float): Anfangstemperatur
        temp_min (float): Minimale Temperatur (Abbruchkriterium)
        cooling_rate (float): Abkühlungsrate pro Iteration
    """
    
    def __init__(
        self,
        cities: List[str],
        distance_matrix: np.ndarray,
        start_city: str,
        temp_initial: float = 10000,
        temp_min: float = 1,
        cooling_rate: float = 0.995
    ):
        """
        Initialisiert Simulated Annealing mit vollständiger Validierung.
        
        Args:
            cities: Liste der Städtenamen
            distance_matrix: NxN Distanzmatrix (symmetrisch)
            start_city: Name der Startstadt
            temp_initial: Anfangstemperatur (höher = mehr Exploration)
            temp_min: Minimale Temperatur (Stopp-Kriterium)
            cooling_rate: Abkühlungsrate (0 < rate < 1)
        
        Raises:
            TypeError: Bei falschen Datentypen
            ValueError: Bei ungültigen Werten
        """
        
        # TYPE VALIDATION
        if not isinstance(cities, list):
            raise TypeError(
                f"'cities' muss eine Liste sein, erhalten: {type(cities).__name__}"
            )
        
        if not isinstance(distance_matrix, np.ndarray):
            raise TypeError(
                f"'distance_matrix' muss np.ndarray sein, "
                f"erhalten: {type(distance_matrix).__name__}"
            )
        
        if not isinstance(start_city, str):
            raise TypeError(
                f"'start_city' muss String sein, "
                f"erhalten: {type(start_city).__name__}"
            )
        
        if not isinstance(temp_initial, (int, float)):
            raise TypeError(
                f"'temp_initial' muss numerisch sein, "
                f"erhalten: {type(temp_initial).__name__}"
            )
        
        if not isinstance(temp_min, (int, float)):
            raise TypeError(
                f"'temp_min' muss numerisch sein, "
                f"erhalten: {type(temp_min).__name__}"
            )
        
        if not isinstance(cooling_rate, (int, float)):
            raise TypeError(
                f"'cooling_rate' muss numerisch sein, "
                f"erhalten: {type(cooling_rate).__name__}"
            )
        
        # BASIC VALUE VALIDATION
        if not cities:
            raise ValueError("Städteliste darf nicht leer sein")
        
        if len(cities) < 3:
            raise ValueError(
                f"Mindestens 3 Städte erforderlich.\n"
                f"Gegeben: {len(cities)} Stadt/Städte"
            )
        
        if len(cities) != len(set(cities)):
            duplicates = [city for city in set(cities) if cities.count(city) > 1]
            raise ValueError(
                f"Städteliste enthält Duplikate: {duplicates}"
            )
        
        # START CITY VALIDATION
        if start_city not in cities:
            similar = [c for c in cities if start_city.lower() in c.lower()]
            error_msg = f"Startstadt '{start_city}' nicht gefunden."
            if similar:
                error_msg += f"\n\nÄhnliche Städte: {', '.join(similar[:3])}"
            raise ValueError(error_msg)
        
        # DISTANCE MATRIX VALIDATION
        if distance_matrix.ndim != 2:
            raise ValueError(
                f"Distanzmatrix muss 2D sein.\n"
                f"Gegeben: {distance_matrix.ndim}D"
            )
        
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError(
                f"Distanzmatrix muss quadratisch sein.\n"
                f"Gegeben: {distance_matrix.shape}"
            )
        
        if len(cities) != distance_matrix.shape[0]:
            raise ValueError(
                f"Anzahl Städte ({len(cities)}) != Matrix-Dimension "
                f"({distance_matrix.shape[0]})"
            )
        
        # MATRIX QUALITY CHECKS
        if np.any(np.isnan(distance_matrix)):
            raise ValueError("Distanzmatrix enthält NaN-Werte")
        
        if np.any(np.isinf(distance_matrix)):
            raise ValueError("Distanzmatrix enthält Inf-Werte")
        
        if np.any(distance_matrix < 0):
            raise ValueError("Distanzmatrix enthält negative Werte")
        
        if not np.allclose(distance_matrix, distance_matrix.T, rtol=1e-5):
            raise ValueError("Distanzmatrix ist nicht symmetrisch")
        
        if not np.allclose(np.diag(distance_matrix), 0, atol=1e-8):
            raise ValueError("Diagonale muss 0 sein")
        
        # TEMPERATURE VALIDATION
        if temp_initial <= 0:
            raise ValueError(
                f"Anfangstemperatur muss positiv sein.\n"
                f"Gegeben: {temp_initial}"
            )
        
        if temp_min <= 0:
            raise ValueError(
                f"Minimale Temperatur muss positiv sein.\n"
                f"Gegeben: {temp_min}"
            )
        
        if temp_min >= temp_initial:
            raise ValueError(
                f"temp_min muss kleiner als temp_initial sein.\n"
                f"temp_initial: {temp_initial}\n"
                f"temp_min: {temp_min}"
            )
        
        # COOLING RATE VALIDATION
        if not (0 < cooling_rate < 1):
            raise ValueError(
                f"Cooling Rate muss zwischen 0 und 1 liegen.\n"
                f"Gegeben: {cooling_rate}\n"
                f"Typische Werte: 0.95 - 0.999"
            )
        
        if cooling_rate < 0.9:
            print(f"⚠️  Warnung: Sehr schnelle Abkühlung ({cooling_rate})")
            print(f"   Algorithmus könnte zu früh konvergieren")
        
        if cooling_rate > 0.999:
            print(f"⚠️  Warnung: Sehr langsame Abkühlung ({cooling_rate})")
            print(f"   Algorithmus könnte sehr lange laufen")
        
        # INITIALIZATION
        self.cities = cities
        self.distance_matrix = distance_matrix
        self.start_idx = cities.index(start_city)
        self.start_city = start_city
        
        self.temp_initial = temp_initial
        self.temp_min = temp_min
        self.cooling_rate = cooling_rate
        
        # Statistics tracking
        self.best_distance_history = []
        self.temperature_history = []
        self.acceptance_history = []
    
    def calculate_route_distance(self, route: List[int]) -> float:
        """
        Berechnet Gesamtdistanz einer Route (inkl. Rückkehr).
        
        Args:
            route: Route als Liste von Stadt-Indices
        
        Returns:
            float: Gesamtdistanz in km
        """
        distance = 0.0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            distance += self.distance_matrix[from_city, to_city]
        return distance
    
    def get_neighbor_2opt(self, route: List[int]) -> List[int]:
        """
        Erzeugt Nachbar-Lösung durch 2-opt swap.
        Vertauscht Teilroute zwischen zwei zufälligen Positionen.
        
        Args:
            route: Aktuelle Route
        
        Returns:
            List[int]: Neue Route (Nachbar)
        """
        new_route = route.copy()
        
        # Wähle zwei verschiedene Positionen (nicht Startstadt)
        i, j = sorted(random.sample(range(1, len(route)), 2))
        
        # Reverse der Teilroute zwischen i und j
        new_route[i:j+1] = reversed(new_route[i:j+1])
        
        return new_route
    
    def acceptance_probability(
        self, 
        old_cost: float, 
        new_cost: float,
        temperature: float
    ) -> float:
        """
        Berechnet Akzeptanzwahrscheinlichkeit nach Metropolis-Kriterium.
        
        Args:
            old_cost: Kosten der aktuellen Lösung
            new_cost: Kosten der neuen Lösung
            temperature: Aktuelle Temperatur
        
        Returns:
            float: Wahrscheinlichkeit (0-1)
        """
        if new_cost < old_cost:
            return 1.0
        return np.exp((old_cost - new_cost) / temperature)
    
    def solve(self, verbose: bool = True) -> Tuple[List[str], float, Dict]:
        """
        Führt Simulated Annealing aus.
        
        Args:
            verbose: Zeige Fortschritt
        
        Returns:
            Tuple[List[str], float, Dict]:
                - Beste Route (Städtenamen)
                - Beste Distanz
                - Statistiken
        """
        start_time = time.time()
        
        if verbose:
            print(f"\n🔥 Simulated Annealing")
            print(f"  Temperatur: {self.temp_initial} → {self.temp_min}")
            print(f"  Abkühlungsrate: {self.cooling_rate}")
        
        # Initiale Lösung: Startstadt fest, Rest zufällig
        current_route = [self.start_idx]
        remaining = [i for i in range(len(self.cities)) 
                    if i != self.start_idx]
        random.shuffle(remaining)
        current_route.extend(remaining)
        
        current_cost = self.calculate_route_distance(current_route)
        best_route = current_route.copy()
        best_cost = current_cost
        initial_cost = current_cost
        
        temperature = self.temp_initial
        iterations = 0
        accepted = 0
        improvements = 0
        
        if verbose:
            print(f"  Initiale Distanz: {current_cost:.2f} km")
        
        # Hauptschleife
        while temperature > self.temp_min:
            iterations += 1
            
            # Nachbar-Lösung generieren
            new_route = self.get_neighbor_2opt(current_route)
            new_cost = self.calculate_route_distance(new_route)
            
            # Akzeptanz-Entscheidung
            accept_prob = self.acceptance_probability(
                current_cost, new_cost, temperature
            )
            
            if random.random() < accept_prob:
                current_route = new_route
                current_cost = new_cost
                accepted += 1
                
                # Beste Lösung aktualisieren
                if new_cost < best_cost:
                    best_route = new_route.copy()
                    best_cost = new_cost
                    improvements += 1
            
            # Tracking (alle 1000 Iterationen)
            if iterations % 1000 == 0:
                self.best_distance_history.append(best_cost)
                self.temperature_history.append(temperature)
                self.acceptance_history.append(accepted / iterations)
                
                if verbose and iterations % 10000 == 0:
                    print(f"  Iteration {iterations:,}: "
                          f"Best={best_cost:.2f}km, "
                          f"Temp={temperature:.1f}, "
                          f"Accept={accepted/iterations:.1%}")
            
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
            'initial_distance': initial_cost,
            'final_distance': best_cost,
            'best_distance_history': self.best_distance_history,
            'temperature_history': self.temperature_history,
            'acceptance_history': self.acceptance_history
        }
        
        if verbose:
            improvement = initial_cost - best_cost
            improvement_pct = (improvement / initial_cost) * 100
            print(f"  Finale Distanz: {best_cost:.2f} km")
            print(f"  Verbesserung: {improvement:.2f} km ({improvement_pct:.1f}%)")
            print(f"  Iterationen: {iterations:,}")
            print(f"  Laufzeit: {runtime:.3f}s")
            print(f"  Akzeptanzrate: {stats['acceptance_rate']:.1%}")
        
        return best_route_names, best_cost, stats