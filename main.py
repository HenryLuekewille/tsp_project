import pandas as pd
import numpy as np
import os
from src.simulated_annealing import SimulatedAnnealingTSP
from src.exact_solver import exact_tsp_brute_force
from src.visualization import (
    visualize_route_on_map,
    visualize_route_static,
    plot_sa_convergence,
    create_comparison_dashboard
)

def compare_algorithms(dataset_name: str = 'top_10_staedte'):
    """Simulated Annealing vs. Exakte Lösung vergleichen"""
    print("=" * 70)
    print(f"TSP VERGLEICH: {dataset_name.upper()}")
    print("=" * 70)
    
    # Daten laden
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_path, 'data', 'test', f'{dataset_name}.csv')
    dist_path = os.path.join(
        base_path, 'data', 'test', f'{dataset_name}_distances.npy'
    )
    
    # Visualisierungsordner erstellen
    vis_dir = os.path.join(base_path, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    try:
        cities_df = pd.read_csv(csv_path)
        distance_matrix = np.load(dist_path)
    except FileNotFoundError as e:
        print(f"❌ Fehler beim Laden: {e}")
        print(f"\n💡 Tipp: Führe zuerst aus:")
        print(f"   python src/data_preprocessing.py")
        return
    
    cities = cities_df['name'].tolist()
    start_city = cities[0]
    
    print(f"\n📍 Anzahl Städte: {len(cities)}")
    print(f"🏁 Startstadt: {start_city}")
    
    # Städteliste anzeigen
    print(f"\n🏙️  DIE {len(cities)} GRÖSSTEN STÄDTE DEUTSCHLANDS:")
    print("-" * 70)
    for i, row in cities_df.iterrows():
        print(f"  {i+1:2d}. {row['name']:<40} "
              f"{int(row['population']):>10,} Einwohner")
    
    # 1. Simulated Annealing
    print(f"\n{'='*70}")
    print("🔥 SIMULATED ANNEALING")
    print("=" * 70)
    
    sa = SimulatedAnnealingTSP(
        cities, distance_matrix, start_city,
        temp_initial=10000,
        temp_min=1,
        cooling_rate=0.995
    )
    sa_route, sa_distance, sa_stats = sa.solve(verbose=True)
    
    # 2. Exakte Lösung
    print(f"\n{'='*70}")
    print("🎯 EXAKTE LÖSUNG (BRUTE FORCE)")
    print("=" * 70)
    
    exact_route = None
    exact_distance = None
    exact_stats = None
    
    try:
        exact_route, exact_distance, exact_stats = exact_tsp_brute_force(
            cities, distance_matrix, start_city, verbose=True
        )
        
        # Vergleich
        print(f"\n{'='*70}")
        print("📊 DETAILLIERTER VERGLEICH")
        print("=" * 70)
        print(f"{'Metrik':<35} {'SA':>15} {'Exakt':>15} {'Differenz':>15}")
        print("-" * 70)
        
        # Distanz
        dist_diff = sa_distance - exact_distance
        print(f"{'Distanz (km)':<35} {sa_distance:>15.2f} "
              f"{exact_distance:>15.2f} "
              f"{dist_diff:>+15.2f}")
        
        # Laufzeit
        time_diff = sa_stats['runtime'] - exact_stats['runtime']
        print(f"{'Laufzeit (s)':<35} {sa_stats['runtime']:>15.3f} "
              f"{exact_stats['runtime']:>15.3f} "
              f"{time_diff:>+15.3f}")
        
        # Speedup
        speedup = exact_stats['runtime'] / sa_stats['runtime']
        if speedup > 1:
            speedup_text = f"SA {speedup:.1f}x schneller"
        else:
            speedup_text = f"BF {1/speedup:.1f}x schneller"
        print(f"{'Speedup':<35} {'':<15} {'':<15} {speedup_text:>15}")
        
        # Qualität
        quality = (exact_distance / sa_distance) * 100
        print(f"{'Qualität (% vom Optimum)':<35} "
              f"{quality:>14.2f}% {'':<15} {'100.00%':>15}")
        
        print("-" * 70)
        
        # Bewertung
        if abs(dist_diff) < 0.01:
            print(f"\n✅ PERFEKT! Simulated Annealing hat die optimale Lösung gefunden!")
        elif quality >= 99.0:
            print(f"\n⭐ HERVORRAGEND! SA ist {100-quality:.2f}% vom Optimum entfernt")
        elif quality >= 95.0:
            print(f"\n👍 GUT! SA ist {100-quality:.2f}% vom Optimum entfernt")
        else:
            print(f"\n⚠️  SA-Lösung ist nur {quality:.2f}% optimal "
                  f"(Differenz: {dist_diff:.2f} km)")
        
    except Exception as e:
        print(f"\n⚠️  Exakte Lösung fehlgeschlagen: {e}")
        print(f"Zeige nur SA-Ergebnis...")
    
    # Route ausgeben
    print(f"\n{'='*70}")
    print(f"🗺️  OPTIMALE ROUTE (SIMULATED ANNEALING)")
    print("=" * 70)
    print(f"Startstadt: {start_city}")
    print("-" * 70)
    
    for i in range(len(sa_route)):
        current_city = sa_route[i]
        next_city = sa_route[(i+1) % len(sa_route)]
        
        current_idx = cities.index(current_city)
        next_idx = cities.index(next_city)
        segment_dist = distance_matrix[current_idx, next_idx]
        
        if i < len(sa_route) - 1:
            print(f"  {i+1:2d}. {current_city:<40} │ "
                  f"→ {segment_dist:>6.1f} km")
        else:
            print(f"  {i+1:2d}. {current_city:<40} │ "
                  f"→ {segment_dist:>6.1f} km")
            print(f"      {'Zurück nach ' + sa_route[0]:<40} │")
    
    print("-" * 70)
    print(f"📏 Gesamtdistanz: {sa_distance:.2f} km")
    print("=" * 70)
    
    # Exakte Route falls vorhanden
    if exact_route:
        print(f"\n{'='*70}")
        print(f"🎯 OPTIMALE ROUTE (EXAKTE LÖSUNG)")
        print("=" * 70)
        print(f"Startstadt: {start_city}")
        print("-" * 70)
        
        for i in range(len(exact_route)):
            current_city = exact_route[i]
            next_city = exact_route[(i+1) % len(exact_route)]
            
            current_idx = cities.index(current_city)
            next_idx = cities.index(next_city)
            segment_dist = distance_matrix[current_idx, next_idx]
            
            if i < len(exact_route) - 1:
                print(f"  {i+1:2d}. {current_city:<40} │ "
                      f"→ {segment_dist:>6.1f} km")
            else:
                print(f"  {i+1:2d}. {current_city:<40} │ "
                      f"→ {segment_dist:>6.1f} km")
                print(f"      {'Zurück nach ' + exact_route[0]:<40} │")
        
        print("-" * 70)
        print(f"📏 Gesamtdistanz: {exact_distance:.2f} km")
        print("=" * 70)
    
    # VISUALISIERUNGEN
    print(f"\n{'='*70}")
    print("🎨 ERSTELLE VISUALISIERUNGEN")
    print("=" * 70)
    
    # 1. Interaktive Karte (SA)
    visualize_route_on_map(
        cities_df, sa_route, sa_distance,
        filename=os.path.join(vis_dir, f'{dataset_name}_sa_route.html'),
        title=f'TSP {dataset_name} - Simulated Annealing'
    )
    
    # 2. Interaktive Karte (Exakt)
    if exact_route:
        visualize_route_on_map(
            cities_df, exact_route, exact_distance,
            filename=os.path.join(vis_dir, f'{dataset_name}_exact_route.html'),
            title=f'TSP {dataset_name} - Optimale Lösung'
        )
    
    # 3. Statischer Plot
    visualize_route_static(
        cities_df, sa_route, sa_distance,
        filename=os.path.join(vis_dir, f'{dataset_name}_route.png')
    )
    
    # 4. SA Konvergenz
    plot_sa_convergence(
        sa_stats,
        filename=os.path.join(vis_dir, f'{dataset_name}_convergence.png')
    )
    
    # 5. Vergleichs-Dashboard
    if exact_route:
        create_comparison_dashboard(
            sa_route, sa_distance, sa_stats,
            exact_route, exact_distance, exact_stats,
            filename=os.path.join(vis_dir, f'{dataset_name}_comparison.png')
        )
    
    print("=" * 70)
    
    # Statistiken
    print(f"\n{'='*70}")
    print(f"📈 ALGORITHMUS-STATISTIKEN")
    print("=" * 70)
    print(f"Simulated Annealing:")
    print(f"  • Iterationen: {sa_stats['iterations']:,}")
    print(f"  • Akzeptierte Züge: {sa_stats['accepted_moves']:,}")
    print(f"  • Verbesserungen: {sa_stats['improvements']:,}")
    print(f"  • Akzeptanzrate: {sa_stats['acceptance_rate']:.1%}")
    print(f"  • Verbesserung: "
          f"{(sa_stats['initial_distance'] - sa_distance):.1f} km "
          f"({((sa_stats['initial_distance'] - sa_distance) / sa_stats['initial_distance'] * 100):.1f}%)")
    print(f"  • Rechenzeit: {sa_stats['runtime']:.3f}s")
    
    if exact_stats:
        print(f"\nExakte Lösung:")
        import math
        perms = exact_stats.get('iterations', math.factorial(len(cities) - 1))
        perms_per_sec = perms / exact_stats['runtime']
        
        print(f"  • Permutationen geprüft: {perms:,}")
        print(f"  • Geschwindigkeit: {perms_per_sec:,.0f} perm/s")
        print(f"  • Rechenzeit: {exact_stats['runtime']:.3f}s")
    
    print("=" * 70)

def main():
    """Hauptprogramm"""
    print("\n" + "=" * 70)
    print("🇩🇪 TSP DEUTSCHLAND - SIMULATED ANNEALING")
    print("=" * 70)
    print("Finde die optimale Rundreise durch die 10 größten Städte")
    print("Vergleich: Simulated Annealing vs. Exakte Lösung (Brute Force)")
    print("=" * 70 + "\n")
    
    compare_algorithms('top_10_staedte')
    
    print("\n" + "=" * 70)
    print("✅ ANALYSE ABGESCHLOSSEN")
    print("=" * 70)
    
    print("\n📊 FAZIT:")
    print("-" * 70)
    print("✅ Simulated Annealing findet gute Lösungen")
    print("✅ SA ist bei großen Problemen deutlich schneller als Brute Force")
    print("⭐ SA skaliert hervorragend zu größeren Probleminstanzen")
    print("🎯 SA bietet guten Trade-off zwischen Qualität und Geschwindigkeit")
    print("-" * 70)
    
    print("\n🎨 VISUALISIERUNGEN:")
    print("-" * 70)
    print("  Erstellt im Ordner: visualizations/")
    print("  • top_10_staedte_sa_route.html (Interaktive SA-Route)")
    print("  • top_10_staedte_exact_route.html (Optimale Route)")
    print("  • top_10_staedte_route.png (Statischer Plot)")
    print("  • top_10_staedte_convergence.png (SA Konvergenz)")
    print("  • top_10_staedte_comparison.png (Vergleichs-Dashboard)")
    print("-" * 70)

if __name__ == "__main__":
    main()