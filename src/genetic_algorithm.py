# src/genetic_algorithm.py
"""
Genetic Algorithm für das Travelling Salesman Problem (TSP)

Implementiert einen genetischen Algorithmus mit vollständigem Error Handling,
um die optimale Route durch eine Menge von Städten zu finden.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import time
import random


class GeneticAlgorithmTSP:
    """
    Genetic Algorithm zur Lösung des Travelling Salesman Problems.
    
    Der Algorithmus verwendet:
    - Tournament Selection für Elternauswahl
    - Ordered Crossover (OX) für Rekombination
    - Swap Mutation für genetische Vielfalt
    - Elitism um beste Lösungen zu bewahren
    
    Attributes:
        cities (List[str]): Liste der Städtenamen
        distance_matrix (np.ndarray): Symmetrische Distanzmatrix
        start_idx (int): Index der Startstadt
        population_size (int): Größe der Population
        generations (int): Maximale Anzahl Generationen
        mutation_rate (float): Wahrscheinlichkeit einer Mutation (0-1)
        elite_size (int): Anzahl Elite-Individuen
        tournament_size (int): Größe des Tournaments bei Selection
    """
    
    def __init__(
        self,
        cities: List[str],
        distance_matrix: np.ndarray,
        start_city: str,
        population_size: int = 100,
        generations: int = 500,
        mutation_rate: float = 0.01,
        elite_size: int = 20,
        tournament_size: int = 5,
        early_stopping_generations: int = 50
    ):
        """
        Initialisiert den Genetic Algorithm mit vollständiger Validierung.
        
        Args:
            cities: Liste der Städtenamen
            distance_matrix: NxN Distanzmatrix (symmetrisch)
            start_city: Name der Startstadt
            population_size: Anzahl Individuen pro Generation
            generations: Maximale Anzahl Generationen
            mutation_rate: Mutationswahrscheinlichkeit (0.0 - 1.0)
            elite_size: Anzahl beste Individuen die überleben
            tournament_size: Größe des Selection-Tournaments
            early_stopping_generations: Stoppe nach N Generationen ohne Verbesserung
        
        Raises:
            TypeError: Bei falschen Datentypen
            ValueError: Bei ungültigen Werten oder Inkonsistenzen
        """
        
        # =====================================================================
        # PHASE 1: TYPE VALIDATION
        # =====================================================================
        
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
        
        if not isinstance(population_size, int):
            raise TypeError(
                f"'population_size' muss int sein, "
                f"erhalten: {type(population_size).__name__}"
            )
        
        if not isinstance(generations, int):
            raise TypeError(
                f"'generations' muss int sein, "
                f"erhalten: {type(generations).__name__}"
            )
        
        if not isinstance(mutation_rate, (int, float)):
            raise TypeError(
                f"'mutation_rate' muss numerisch sein, "
                f"erhalten: {type(mutation_rate).__name__}"
            )
        
        # =====================================================================
        # PHASE 2: BASIC VALUE VALIDATION
        # =====================================================================
        
        if not cities:
            raise ValueError(
                "Städteliste darf nicht leer sein"
            )
        
        if len(cities) < 3:
            raise ValueError(
                f"Mindestens 3 Städte erforderlich für TSP.\n"
                f"Gegeben: {len(cities)} Stadt/Städte.\n"
                f"Mit <3 Städten ist TSP trivial."
            )
        
        if len(cities) > 1000:
            raise ValueError(
                f"Zu viele Städte für diesen Algorithmus.\n"
                f"Gegeben: {len(cities)} Städte\n"
                f"Maximum: 1000 Städte\n"
                f"Tipp: Verwenden Sie einen spezialisierten TSP-Solver für große Instanzen."
            )
        
        # Check for duplicate cities
        if len(cities) != len(set(cities)):
            duplicates = [city for city in set(cities) if cities.count(city) > 1]
            raise ValueError(
                f"Städteliste enthält Duplikate: {duplicates}\n"
                f"Jede Stadt darf nur einmal vorkommen."
            )
        
        # =====================================================================
        # PHASE 3: START CITY VALIDATION
        # =====================================================================
        
        if start_city not in cities:
            # Hilfreicher Fehler: Suche ähnliche Städtenamen
            similar_cities = [
                city for city in cities 
                if start_city.lower() in city.lower()
            ]
            
            error_msg = f"Startstadt '{start_city}' nicht in Städteliste gefunden."
            
            if similar_cities:
                error_msg += f"\n\nMeinten Sie eine dieser Städte?"
                for city in similar_cities[:3]:
                    error_msg += f"\n  • {city}"
            else:
                error_msg += f"\n\nVerfügbare Städte (erste 10):"
                for city in cities[:10]:
                    error_msg += f"\n  • {city}"
                if len(cities) > 10:
                    error_msg += f"\n  ... und {len(cities) - 10} weitere"
            
            raise ValueError(error_msg)
        
        # =====================================================================
        # PHASE 4: DISTANCE MATRIX VALIDATION
        # =====================================================================
        
        # Check dimensions
        if distance_matrix.ndim != 2:
            raise ValueError(
                f"Distanzmatrix muss 2-dimensional sein.\n"
                f"Gegeben: {distance_matrix.ndim}D mit Shape {distance_matrix.shape}"
            )
        
        # Check square
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError(
                f"Distanzmatrix muss quadratisch sein.\n"
                f"Gegeben: {distance_matrix.shape[0]}x{distance_matrix.shape[1]}"
            )
        
        # Check size matches cities
        if len(cities) != distance_matrix.shape[0]:
            raise ValueError(
                f"Anzahl Städte stimmt nicht mit Matrix-Dimension überein.\n"
                f"Städte: {len(cities)}\n"
                f"Matrix: {distance_matrix.shape[0]}x{distance_matrix.shape[1]}\n"
                f"Beide müssen identisch sein."
            )
        
        # =====================================================================
        # PHASE 5: MATRIX DATA QUALITY CHECKS
        # =====================================================================
        
        # Check for NaN values
        if np.any(np.isnan(distance_matrix)):
            nan_positions = np.argwhere(np.isnan(distance_matrix))
            raise ValueError(
                f"Distanzmatrix enthält NaN-Werte an {len(nan_positions)} Positionen.\n"
                f"Erste NaN-Position: [{nan_positions[0][0]}, {nan_positions[0][1]}]\n"
                f"Alle Distanzen müssen numerische Werte sein."
            )
        
        # Check for infinite values
        if np.any(np.isinf(distance_matrix)):
            inf_positions = np.argwhere(np.isinf(distance_matrix))
            raise ValueError(
                f"Distanzmatrix enthält Inf-Werte an {len(inf_positions)} Positionen.\n"
                f"Erste Inf-Position: [{inf_positions[0][0]}, {inf_positions[0][1]}]\n"
                f"Alle Distanzen müssen endlich sein."
            )
        
        # Check for negative values
        if np.any(distance_matrix < 0):
            neg_positions = np.argwhere(distance_matrix < 0)
            raise ValueError(
                f"Distanzmatrix enthält negative Werte.\n"
                f"Gefunden: {len(neg_positions)} negative Einträge\n"
                f"Erste negative Position: [{neg_positions[0][0]}, {neg_positions[0][1]}] "
                f"= {distance_matrix[neg_positions[0][0], neg_positions[0][1]]:.2f}\n"
                f"Distanzen müssen nicht-negativ sein."
            )
        
        # Check symmetry
        if not np.allclose(distance_matrix, distance_matrix.T, rtol=1e-5, atol=1e-8):
            # Find asymmetric entries
            diff = np.abs(distance_matrix - distance_matrix.T)
            asymmetric = np.argwhere(diff > 1e-5)
            
            if len(asymmetric) > 0:
                i, j = asymmetric[0]
                raise ValueError(
                    f"Distanzmatrix ist nicht symmetrisch.\n"
                    f"TSP erfordert symmetrische Distanzen (d(A,B) = d(B,A)).\n"
                    f"Beispiel-Asymmetrie bei Position [{i}, {j}]:\n"
                    f"  Matrix[{i}, {j}] = {distance_matrix[i, j]:.6f}\n"
                    f"  Matrix[{j}, {i}] = {distance_matrix[j, i]:.6f}\n"
                    f"  Differenz: {abs(distance_matrix[i, j] - distance_matrix[j, i]):.6f}"
                )
        
        # Check diagonal is zero
        diagonal = np.diag(distance_matrix)
        if not np.allclose(diagonal, 0, atol=1e-8):
            non_zero_diag = np.argwhere(np.abs(diagonal) > 1e-8)
            raise ValueError(
                f"Diagonale der Matrix muss 0 sein (Distanz von Stadt zu sich selbst).\n"
                f"Gefunden: {len(non_zero_diag)} nicht-null Einträge auf Diagonale\n"
                f"Beispiel: Position [{non_zero_diag[0][0]}, {non_zero_diag[0][0]}] "
                f"= {diagonal[non_zero_diag[0][0]]:.6f}"
            )
        
        # Check triangle inequality (optional but recommended)
        # Nur Stichproben-Check für große Matrizen (zu aufwändig sonst)
        if len(cities) <= 50:
            n = len(cities)
            violations = []
            for i in range(min(n, 10)):
                for j in range(i+1, min(n, 10)):
                    for k in range(j+1, min(n, 10)):
                        d_ij = distance_matrix[i, j]
                        d_jk = distance_matrix[j, k]
                        d_ik = distance_matrix[i, k]
                        
                        if d_ik > d_ij + d_jk + 1e-6:  # Kleine Toleranz
                            violations.append((i, j, k, d_ik, d_ij + d_jk))
            
            if violations:
                i, j, k, d_ik, d_ij_jk = violations[0]
                print(f"⚠️  Warnung: Dreiecksungleichung verletzt bei Städten "
                      f"{i}, {j}, {k}")
                print(f"   d({i},{k}) = {d_ik:.2f} > "
                      f"d({i},{j}) + d({j},{k}) = {d_ij_jk:.2f}")
                print(f"   Matrix könnte inkonsistent sein (z.B. durch Rundungsfehler)")
        
        # =====================================================================
        # PHASE 6: PARAMETER VALIDATION
        # =====================================================================
        
        if population_size < 10:
            raise ValueError(
                f"Population Size muss mindestens 10 sein.\n"
                f"Gegeben: {population_size}\n"
                f"Kleinere Populationen können nicht ausreichend diversifizieren."
            )
        
        if population_size > 10000:
            raise ValueError(
                f"Population Size zu groß: {population_size}\n"
                f"Maximum: 10000\n"
                f"Größere Populationen führen zu extremen Rechenzeiten."
            )
        
        if generations < 1:
            raise ValueError(
                f"Generations muss mindestens 1 sein.\n"
                f"Gegeben: {generations}"
            )
        
        if generations > 100000:
            raise ValueError(
                f"Generations zu groß: {generations}\n"
                f"Maximum: 100000\n"
                f"Tipp: Verwenden Sie Early Stopping statt vieler Generationen."
            )
        
        if not (0 <= mutation_rate <= 1):
            raise ValueError(
                f"Mutation Rate muss zwischen 0 und 1 liegen.\n"
                f"Gegeben: {mutation_rate}\n"
                f"0 = keine Mutation, 1 = immer Mutation\n"
                f"Typische Werte: 0.001 - 0.1"
            )
        
        if elite_size < 0:
            raise ValueError(
                f"Elite Size muss nicht-negativ sein.\n"
                f"Gegeben: {elite_size}"
            )
        
        if elite_size >= population_size:
            raise ValueError(
                f"Elite Size muss kleiner als Population Size sein.\n"
                f"Elite Size: {elite_size}\n"
                f"Population Size: {population_size}\n"
                f"Sonst gibt es keine Evolution (alle überleben automatisch)."
            )
        
        if tournament_size < 2:
            raise ValueError(
                f"Tournament Size muss mindestens 2 sein.\n"
                f"Gegeben: {tournament_size}\n"
                f"Tournaments mit <2 Teilnehmern ergeben keinen Sinn."
            )
        
        if tournament_size > population_size:
            raise ValueError(
                f"Tournament Size kann nicht größer als Population Size sein.\n"
                f"Tournament Size: {tournament_size}\n"
                f"Population Size: {population_size}"
            )
        
        if early_stopping_generations < 1:
            raise ValueError(
                f"Early Stopping Generations muss mindestens 1 sein.\n"
                f"Gegeben: {early_stopping_generations}"
            )
        
        # =====================================================================
        # PHASE 7: INITIALIZATION (All checks passed!)
        # =====================================================================
        
        self.cities = cities
        self.distance_matrix = distance_matrix
        self.start_idx = cities.index(start_city)
        self.start_city = start_city
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.early_stopping_generations = early_stopping_generations
        
        # Statistics tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
        
        # Best solution found
        self.best_route = None
        self.best_distance = float('inf')
    
    def create_route(self) -> np.ndarray:
        """
        Erstellt eine zufällige Route (Permutation der Städte außer Start).
        
        Returns:
            np.ndarray: Route als Array von Indices
        """
        # Alle Städte außer Startstadt
        cities_to_visit = list(range(len(self.cities)))
        cities_to_visit.remove(self.start_idx)
        
        # Zufällig mischen
        np.random.shuffle(cities_to_visit)
        
        # Startstadt an den Anfang
        route = np.array([self.start_idx] + cities_to_visit)
        return route
    
    def calculate_route_distance(self, route: np.ndarray) -> float:
        """
        Berechnet die Gesamtdistanz einer Route.
        
        Args:
            route: Route als Array von Stadt-Indices
        
        Returns:
            float: Gesamtdistanz der Rundreise
        """
        distance = 0.0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]  # Zurück zum Start
            distance += self.distance_matrix[from_city, to_city]
        return distance
    
    def fitness(self, route: np.ndarray) -> float:
        """
        Berechnet Fitness (1 / Distanz, größer ist besser).
        
        Args:
            route: Route als Array von Stadt-Indices
        
        Returns:
            float: Fitness-Wert
        """
        distance = self.calculate_route_distance(route)
        return 1.0 / distance if distance > 0 else 0.0
    
    def create_initial_population(self) -> List[np.ndarray]:
        """
        Erstellt initiale Population von zufälligen Routen.
        
        Returns:
            List[np.ndarray]: Liste von Routen
        """
        return [self.create_route() for _ in range(self.population_size)]
    
    def tournament_selection(
        self, 
        population: List[np.ndarray], 
        fitnesses: np.ndarray
    ) -> np.ndarray:
        """
        Wählt ein Individuum via Tournament Selection.
        
        Args:
            population: Aktuelle Population
            fitnesses: Fitness-Werte aller Individuen
        
        Returns:
            np.ndarray: Gewinner-Route
        """
        # Wähle zufällig tournament_size Individuen
        tournament_indices = np.random.choice(
            len(population), 
            size=self.tournament_size, 
            replace=False
        )
        
        # Finde bestes Individuum im Tournament
        tournament_fitnesses = fitnesses[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        
        return population[winner_idx].copy()
    
    def ordered_crossover(
        self, 
        parent1: np.ndarray, 
        parent2: np.ndarray
    ) -> np.ndarray:
        """
        Ordered Crossover (OX) - bewahrt relative Reihenfolge.
        
        Args:
            parent1: Erste Elternroute
            parent2: Zweite Elternroute
        
        Returns:
            np.ndarray: Kind-Route
        """
        size = len(parent1)
        
        # Wähle zufälliges Segment
        start, end = sorted(np.random.choice(range(1, size), size=2, replace=False))
        
        # Kopiere Segment von Parent1
        child = np.full(size, -1, dtype=int)
        child[start:end] = parent1[start:end]
        
        # Fülle Rest mit Parent2 in Reihenfolge
        child_idx = end
        for city in parent2:
            if city not in child:
                if child_idx >= size:
                    child_idx = 1  # Überspringe Startstadt (Index 0)
                if child_idx == start:
                    child_idx = end
                child[child_idx] = city
                child_idx += 1
        
        # Stelle sicher dass Startstadt am Anfang ist
        child[0] = self.start_idx
        
        return child
    
    def swap_mutation(self, route: np.ndarray) -> np.ndarray:
        """
        Swap Mutation - tauscht zwei zufällige Städte (nicht Start).
        
        Args:
            route: Zu mutierende Route
        
        Returns:
            np.ndarray: Mutierte Route
        """
        mutated = route.copy()
        
        # Wähle zwei verschiedene Positionen (nicht Position 0 = Start)
        idx1, idx2 = np.random.choice(range(1, len(route)), size=2, replace=False)
        
        # Tausche
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        return mutated
    
    def calculate_diversity(self, population: List[np.ndarray]) -> float:
        """
        Berechnet genetische Diversität (Anzahl einzigartiger Routen).
        
        Args:
            population: Aktuelle Population
        
        Returns:
            float: Diversität als Anzahl einzigartiger Routen
        """
        unique_routes = set()
        for route in population:
            # Konvertiere zu Tuple für Hashing
            route_tuple = tuple(route)
            unique_routes.add(route_tuple)
        
        return len(unique_routes)
    
    def solve(self, verbose: bool = False) -> Tuple[List[str], float, Dict]:
        """
        Führt Genetic Algorithm aus.
        
        Args:
            verbose: Zeige Fortschritt während Berechnung
        
        Returns:
            Tuple[List[str], float, Dict]: 
                - Beste Route (Städtenamen)
                - Beste Distanz
                - Statistiken
        """
        start_time = time.time()
        
        if verbose:
            print(f"\n🧬 Genetic Algorithm")
            print(f"  Population: {self.population_size}")
            print(f"  Generationen: {self.generations}")
            print(f"  Mutation Rate: {self.mutation_rate*100:.1f}%")
            print(f"  Elite Size: {self.elite_size}")
        
        # Erstelle initiale Population
        population = self.create_initial_population()
        
        generations_without_improvement = 0
        initial_best_distance = None
        
        for generation in range(self.generations):
            # Berechne Fitness für alle Individuen
            fitnesses = np.array([self.fitness(route) for route in population])
            
            # Finde beste Route dieser Generation
            best_idx = np.argmax(fitnesses)
            best_route = population[best_idx]
            best_distance = self.calculate_route_distance(best_route)
            
            # Tracking
            avg_distance = np.mean([self.calculate_route_distance(r) for r in population])
            diversity = self.calculate_diversity(population)
            
            self.best_fitness_history.append(best_distance)
            self.avg_fitness_history.append(avg_distance)
            self.diversity_history.append(diversity)
            
            # Speichere initiale beste Distanz
            if generation == 0:
                initial_best_distance = best_distance
            
            # Ausgabe
            if verbose and (generation % 50 == 0 or generation == self.generations - 1):
                print(f"  Gen {generation:4d}: "
                      f"Best={best_distance:.2f}km, "
                      f"Avg={avg_distance:.2f}km, "
                      f"Diversity={diversity:.1f}")
            
            # Update global best
            if best_distance < self.best_distance:
                self.best_distance = best_distance
                self.best_route = best_route.copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            # Early Stopping
            if generations_without_improvement >= self.early_stopping_generations:
                if verbose:
                    print(f"  Early stopping nach {generation + 1} Generationen")
                break
            
            # Erstelle neue Population
            new_population = []
            
            # Elitism: Kopiere beste Individuen
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Fülle Rest durch Selection, Crossover, Mutation
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                # Crossover
                child = self.ordered_crossover(parent1, parent2)
                
                # Mutation
                if np.random.random() < self.mutation_rate:
                    child = self.swap_mutation(child)
                
                new_population.append(child)
            
            population = new_population
        
        # Finale Ausgabe
        end_time = time.time()
        runtime = end_time - start_time
        
        improvement = ((initial_best_distance - self.best_distance) / 
                      initial_best_distance * 100) if initial_best_distance else 0
        
        if verbose:
            print(f"  Finale Distanz: {self.best_distance:.2f} km")
            print(f"  Verbesserung: {improvement:.1f}%")
            print(f"  Laufzeit: {runtime:.3f}s")
        
        # Konvertiere Route zu Städtenamen
        route_names = [self.cities[idx] for idx in self.best_route]
        
        # Statistiken
        stats = {
            'runtime': runtime,
            'generations': generation + 1,
            'improvement': improvement,
            'initial_distance': initial_best_distance,
            'final_distance': self.best_distance,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'diversity_history': self.diversity_history
        }
        
        return route_names, self.best_distance, stats