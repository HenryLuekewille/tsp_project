# src/visualization.py
"""
Visualisierungen für TSP-Lösungen
"""

import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import os


def visualize_route_on_map(
    cities_df: pd.DataFrame,
    route: List[str],
    distance: float,
    filename: str = "tsp_route.html",
    title: str = "TSP Route"
):
    """
    Erstellt interaktive Karte mit TSP-Route
    
    Args:
        cities_df: DataFrame mit Städten (columns: name, latitude, longitude)
        route: Liste von Städtenamen in Besuchsreihenfolge
        distance: Gesamtdistanz der Route
        filename: Output HTML-Datei
        title: Titel der Karte
    """
    # Zentrum von Deutschland
    map_center = [51.1657, 10.4515]
    m = folium.Map(location=map_center, zoom_start=6, 
                   tiles='OpenStreetMap')
    
    # Städte als Dict für schnellen Zugriff
    city_coords = {}
    for _, row in cities_df.iterrows():
        city_coords[row['name']] = (row['latitude'], row['longitude'])
    
    # Route zeichnen
    route_coords = [city_coords[city] for city in route]
    route_coords.append(route_coords[0])  # Zurück zum Start
    
    folium.PolyLine(
        route_coords,
        color='red',
        weight=3,
        opacity=0.8,
        popup=f'Route: {distance:.2f} km'
    ).add_to(m)
    
    # Städte markieren
    for i, city in enumerate(route, 1):
        lat, lon = city_coords[city]
        
        # Startstadt speziell markieren
        if i == 1:
            icon = folium.Icon(color='green', icon='play')
            popup_text = f'🏁 START: {city}'
        else:
            icon = folium.Icon(color='blue', icon='info-sign')
            popup_text = f'{i}. {city}'
        
        folium.Marker(
            [lat, lon],
            popup=popup_text,
            tooltip=city,
            icon=icon
        ).add_to(m)
    
    # Titel hinzufügen
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 300px; height: 90px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
        <b>{title}</b><br>
        Städte: {len(route)}<br>
        Gesamtdistanz: {distance:.2f} km
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Speichern
    m.save(filename)
    print(f"  ✓ Interaktive Karte: {filename}")
    return m


def visualize_route_static(
    cities_df: pd.DataFrame,
    route: List[str],
    distance: float,
    filename: str = "tsp_route.png"
):
    """Statisches Plot der Route"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Städte
    city_coords = {}
    for _, row in cities_df.iterrows():
        city_coords[row['name']] = (row['longitude'], row['latitude'])
    
    # Route zeichnen
    route_coords = [city_coords[city] for city in route]
    route_coords.append(route_coords[0])
    
    lons, lats = zip(*route_coords)
    ax.plot(lons, lats, 'r-', linewidth=2, alpha=0.7, label='Route')
    
    # Städte markieren
    for i, city in enumerate(route, 1):
        lon, lat = city_coords[city]
        
        if i == 1:
            ax.plot(lon, lat, 'go', markersize=15, label='Start')
        else:
            ax.plot(lon, lat, 'bo', markersize=10)
        
        # Stadtnamen (nur Hauptname ohne Zusätze)
        city_short = city.split(",")[0]
        ax.annotate(f'{i}. {city_short}', 
                   (lon, lat), 
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=8)
    
    ax.set_xlabel('Längengrad')
    ax.set_ylabel('Breitengrad')
    ax.set_title(f'TSP Route - {distance:.2f} km', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Statischer Plot: {filename}")
    plt.close()


def plot_ga_convergence(ga_stats: dict, filename: str = "ga_convergence.png"):
    """Zeigt Konvergenz des Genetic Algorithm"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Beste Fitness über Zeit
    ax1 = axes[0, 0]
    ax1.plot(ga_stats['best_fitness_history'], 'b-', linewidth=2)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Beste Distanz (km)')
    ax1.set_title('Konvergenz: Beste Lösung')
    ax1.grid(True, alpha=0.3)
    
    # 2. Durchschnittliche Fitness
    ax2 = axes[0, 1]
    ax2.plot(ga_stats['avg_fitness_history'], 'g-', linewidth=2, 
             label='Durchschnitt')
    ax2.plot(ga_stats['best_fitness_history'], 'b--', linewidth=1, 
             label='Beste')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Distanz (km)')
    ax2.set_title('Population Fitness')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Diversität
    ax3 = axes[1, 0]
    ax3.plot(ga_stats['diversity_history'], 'r-', linewidth=2)
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Anzahl unique Routes')
    ax3.set_title('Genetische Diversität')
    ax3.grid(True, alpha=0.3)
    
    # 4. Verbesserung pro Generation
    ax4 = axes[1, 1]
    improvements = np.diff(ga_stats['best_fitness_history'])
    ax4.bar(range(len(improvements)), -improvements, alpha=0.7)
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Verbesserung (km)')
    ax4.set_title('Verbesserung pro Generation')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Konvergenz-Plot: {filename}")
    plt.close()


def create_comparison_dashboard(
    ga_route: List[str],
    ga_distance: float,
    ga_stats: dict,
    exact_route: List[str],
    exact_distance: float,
    exact_stats: dict,
    filename: str = "comparison_dashboard.png"
):
    """Erstellt Vergleichs-Dashboard GA vs. Exakt"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Titel
    fig.suptitle('TSP Algorithmen-Vergleich: GA vs. Brute Force', 
                 fontsize=16, fontweight='bold')
    
    # 1. Distanz-Vergleich (Balken)
    ax1 = fig.add_subplot(gs[0, 0])
    algorithms = ['GA', 'Exakt']
    distances = [ga_distance, exact_distance]
    colors = ['lightblue', 'lightgreen']
    bars = ax1.bar(algorithms, distances, color=colors, edgecolor='black')
    ax1.set_ylabel('Distanz (km)')
    ax1.set_title('Routenlänge')
    
    # Werte auf Balken
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # 2. Laufzeit-Vergleich (Log-Skala)
    ax2 = fig.add_subplot(gs[0, 1])
    runtimes = [ga_stats['runtime'], exact_stats['runtime']]
    bars = ax2.bar(algorithms, runtimes, color=colors, edgecolor='black')
    ax2.set_ylabel('Laufzeit (s)')
    ax2.set_title('Berechnungszeit')
    ax2.set_yscale('log')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s',
                ha='center', va='bottom')
    
    # 3. Qualitäts-Metrik
    ax3 = fig.add_subplot(gs[0, 2])
    quality = (exact_distance / ga_distance) * 100
    ax3.text(0.5, 0.5, f'{quality:.2f}%', 
            ha='center', va='center', fontsize=48, fontweight='bold')
    ax3.text(0.5, 0.2, 'GA Qualität\n(% vom Optimum)', 
            ha='center', va='center', fontsize=12)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # 4. GA Konvergenz
    ax4 = fig.add_subplot(gs[1, :])
    ax4.plot(ga_stats['best_fitness_history'], 'b-', linewidth=2)
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Beste Distanz (km)')
    ax4.set_title('GA Konvergenz über Generationen')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=exact_distance, color='g', linestyle='--', 
                label='Optimum (Brute Force)')
    ax4.legend()
    
    # 5. Statistik-Tabelle
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Sichere Extraktion von Iterationen
    import math
    exact_iterations = exact_stats.get('iterations', 
                                       math.factorial(len(ga_route) - 1))
    
    stats_data = [
        ['Metrik', 'GA', 'Brute Force', 'Differenz'],
        ['Distanz (km)', f'{ga_distance:.2f}', f'{exact_distance:.2f}', 
         f'{ga_distance - exact_distance:+.2f}'],
        ['Laufzeit (s)', f'{ga_stats["runtime"]:.3f}', 
         f'{exact_stats["runtime"]:.3f}',
         f'{ga_stats["runtime"] - exact_stats["runtime"]:+.3f}'],
        ['Iterationen', f'{ga_stats["generations"]:,}', 
         f'{exact_iterations:,}', '-'],
        ['Qualität', f'{quality:.2f}%', '100.00%', 
         f'{quality - 100:+.2f}%']
    ]
    
    table = ax5.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header-Zeile hervorheben
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Vergleichs-Dashboard: {filename}")
    plt.close()