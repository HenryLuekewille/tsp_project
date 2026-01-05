"""
Visualisierung für TSP-Lösungen

Bietet verschiedene Visualisierungsoptionen:
- Interaktive Karten mit folium
- Statische Plots mit matplotlib
- Konvergenz-Plots
- Vergleichs-Dashboards
"""

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


def _get_lat_lon_columns(cities_df: pd.DataFrame) -> tuple:
    """
    Findet die Spalten für Latitude und Longitude.
    
    Args:
        cities_df: DataFrame mit Städtedaten
    
    Returns:
        tuple: (lat_column, lon_column)
    
    Raises:
        ValueError: Wenn Spalten nicht gefunden werden
    """
    columns = cities_df.columns.tolist()
    
    # Mögliche Spaltennamen für Latitude
    lat_names = ['lat', 'latitude', 'Lat', 'Latitude', 'breitengrad', 'Breitengrad']
    lon_names = ['lon', 'longitude', 'Lon', 'Longitude', 'laengengrad', 
                 'Laengengrad', 'längengrad', 'Längengrad']
    
    lat_col = None
    lon_col = None
    
    # Suche nach Latitude-Spalte
    for name in lat_names:
        if name in columns:
            lat_col = name
            break
    
    # Suche nach Longitude-Spalte
    for name in lon_names:
        if name in columns:
            lon_col = name
            break
    
    if lat_col is None or lon_col is None:
        raise ValueError(
            f"Konnte Latitude/Longitude Spalten nicht finden.\n"
            f"Verfügbare Spalten: {columns}\n"
            f"Erwartet: Eine von {lat_names} und eine von {lon_names}"
        )
    
    return lat_col, lon_col


def visualize_route_on_map(
    cities_df: pd.DataFrame,
    route: List[str],
    distance: float,
    filename: str = "route_map.html",
    title: str = "TSP Route"
) -> None:
    """
    Erstellt interaktive Karte mit folium.
    
    Args:
        cities_df: DataFrame mit Städten (name, lat/latitude, lon/longitude, population)
        route: Route als Liste von Städtenamen
        distance: Gesamtdistanz der Route
        filename: Ausgabedatei (.html)
        title: Titel der Karte
    """
    # Finde Spalten für Lat/Lon
    lat_col, lon_col = _get_lat_lon_columns(cities_df)
    
    # Zentrum der Karte berechnen
    center_lat = cities_df[lat_col].mean()
    center_lon = cities_df[lon_col].mean()
    
    # Karte erstellen
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Titel hinzufügen
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; 
                left: 50px; 
                width: 400px; 
                height: 90px; 
                background-color: white;
                border: 2px solid grey;
                z-index: 9999;
                font-size: 16px;
                padding: 10px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
                ">
        <h3 style="margin: 0; color: #2c3e50;">{title}</h3>
        <p style="margin: 5px 0;"><strong>Gesamtdistanz:</strong> 
           {distance:.2f} km</p>
        <p style="margin: 5px 0;"><strong>Anzahl Städte:</strong> 
           {len(route)}</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Städte markieren
    for idx, city in enumerate(route):
        city_data = cities_df[cities_df['name'] == city].iloc[0]
        
        # Farbe je nach Position
        if idx == 0:
            color = 'green'
            icon = 'play'
            popup_prefix = '🏁 START'
        elif idx == len(route) - 1:
            color = 'red'
            icon = 'stop'
            popup_prefix = '🏁 ZIEL'
        else:
            color = 'blue'
            icon = 'info-sign'
            popup_prefix = f'#{idx}'
        
        # Popup-Inhalt
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 200px;">
            <h4 style="margin: 0; color: {color};">{popup_prefix}: {city}</h4>
            <hr style="margin: 5px 0;">
            <p style="margin: 3px 0;">
                <strong>Einwohner:</strong> 
                {int(city_data['population']):,}
            </p>
            <p style="margin: 3px 0;">
                <strong>Position:</strong> {idx + 1} / {len(route)}
            </p>
            <p style="margin: 3px 0; font-size: 11px; color: #666;">
                Lat: {city_data[lat_col]:.4f}°<br>
                Lon: {city_data[lon_col]:.4f}°
            </p>
        </div>
        """
        
        folium.Marker(
            location=[city_data[lat_col], city_data[lon_col]],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{idx + 1}. {city}",
            icon=folium.Icon(color=color, icon=icon)
        ).add_to(m)
    
    # Route als Linien zeichnen
    route_coords = []
    for city in route:
        city_data = cities_df[cities_df['name'] == city].iloc[0]
        route_coords.append([city_data[lat_col], city_data[lon_col]])
    
    # Zurück zum Start
    route_coords.append(route_coords[0])
    
    folium.PolyLine(
        route_coords,
        color='darkblue',
        weight=3,
        opacity=0.7,
        popup=f"Gesamtroute: {distance:.2f} km"
    ).add_to(m)
    
    # Karte speichern
    m.save(filename)
    print(f"  ✓ Interaktive Karte: {filename}")


def visualize_route_static(
    cities_df: pd.DataFrame,
    route: List[str],
    distance: float,
    filename: str = "route_static.png"
) -> None:
    """
    Erstellt statischen Plot der Route mit matplotlib.
    
    Args:
        cities_df: DataFrame mit Städten
        route: Route als Liste von Städtenamen
        distance: Gesamtdistanz
        filename: Ausgabedatei (.png)
    """
    # Finde Spalten für Lat/Lon
    lat_col, lon_col = _get_lat_lon_columns(cities_df)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Deutschland-ähnliche Hintergrundfarbe
    ax.set_facecolor('#f0f0f0')
    
    # Route zeichnen
    route_coords = []
    for city in route:
        city_data = cities_df[cities_df['name'] == city].iloc[0]
        route_coords.append([city_data[lon_col], city_data[lat_col]])
    
    # Zurück zum Start
    route_coords.append(route_coords[0])
    route_coords = np.array(route_coords)
    
    # Linien
    ax.plot(
        route_coords[:, 0],
        route_coords[:, 1],
        'b-',
        linewidth=2,
        alpha=0.6,
        zorder=1
    )
    
    # Punkte mit Größe nach Bevölkerung
    for idx, city in enumerate(route):
        city_data = cities_df[cities_df['name'] == city].iloc[0]
        
        # Punktgröße basiert auf Bevölkerung
        size = 100 + (city_data['population'] / 10000)
        
        # Farbe
        if idx == 0:
            color = 'green'
            marker = 's'  # Square für Start
            label = 'Start'
        else:
            color = 'red'
            marker = 'o'
            label = None
        
        ax.scatter(
            city_data[lon_col],
            city_data[lat_col],
            s=size,
            c=color,
            marker=marker,
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5,
            zorder=3,
            label=label
        )
        
        # Stadt-Labels
        ax.annotate(
            f"{idx + 1}. {city}",
            xy=(city_data[lon_col], city_data[lat_col]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='white',
                edgecolor='gray',
                alpha=0.8
            ),
            zorder=4
        )
    
    # Titel und Labels
    ax.set_title(
        f'TSP Route durch {len(route)} Städte\n'
        f'Gesamtdistanz: {distance:.2f} km',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Längengrad', fontsize=12)
    ax.set_ylabel('Breitengrad', fontsize=12)
    
    # Legende
    ax.legend(loc='upper right', fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Statischer Plot: {filename}")
    plt.close()


def plot_sa_convergence(
    sa_stats: Dict,
    filename: str = "sa_convergence.png"
) -> None:
    """
    Zeigt Konvergenz von Simulated Annealing.
    
    Args:
        sa_stats: Statistiken vom SA-Algorithmus
        filename: Ausgabedatei
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Beste Distanz über Zeit
    ax1 = axes[0, 0]
    history = sa_stats['best_distance_history']
    if history:
        iterations = np.arange(len(history)) * 1000
        ax1.plot(iterations, history, 'b-', linewidth=2, alpha=0.8)
        ax1.fill_between(
            iterations,
            history,
            alpha=0.3,
            color='blue'
        )
        ax1.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Beste Distanz (km)', fontsize=11, fontweight='bold')
        ax1.set_title(
            'Konvergenz der besten Lösung',
            fontsize=13,
            fontweight='bold'
        )
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Verbesserung annotieren
        initial = history[0]
        final = history[-1]
        improvement = initial - final
        ax1.annotate(
            f'Verbesserung:\n{improvement:.1f} km\n'
            f'({improvement/initial*100:.1f}%)',
            xy=(iterations[-1], final),
            xytext=(-120, 40),
            textcoords='offset points',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            fontsize=9,
            arrowprops=dict(arrowstyle='->', color='green', lw=2)
        )
    
    # 2. Temperatur über Zeit
    ax2 = axes[0, 1]
    temp_history = sa_stats['temperature_history']
    if temp_history:
        iterations = np.arange(len(temp_history)) * 1000
        ax2.plot(iterations, temp_history, 'r-', linewidth=2, alpha=0.8)
        ax2.fill_between(
            iterations,
            temp_history,
            alpha=0.3,
            color='red'
        )
        ax2.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Temperatur', fontsize=11, fontweight='bold')
        ax2.set_title(
            'Abkühlungskurve',
            fontsize=13,
            fontweight='bold'
        )
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Temperaturbereich annotieren
        ax2.axhline(
            y=sa_stats['temperature_history'][0],
            color='orange',
            linestyle='--',
            linewidth=2,
            alpha=0.5,
            label=f"Start: {sa_stats['temperature_history'][0]:.0f}"
        )
        ax2.axhline(
            y=sa_stats['temperature_history'][-1],
            color='blue',
            linestyle='--',
            linewidth=2,
            alpha=0.5,
            label=f"Ende: {sa_stats['temperature_history'][-1]:.2f}"
        )
        ax2.legend(loc='upper right', fontsize=9)
    
    # 3. Akzeptanzrate über Zeit
    ax3 = axes[1, 0]
    accept_history = sa_stats['acceptance_history']
    if accept_history:
        iterations = np.arange(len(accept_history)) * 1000
        ax3.plot(iterations, accept_history, 'g-', linewidth=2, alpha=0.8)
        ax3.fill_between(
            iterations,
            accept_history,
            alpha=0.3,
            color='green'
        )
        ax3.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Akzeptanzrate', fontsize=11, fontweight='bold')
        ax3.set_title(
            'Akzeptanzrate über Zeit',
            fontsize=13,
            fontweight='bold'
        )
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # 50% Linie
        ax3.axhline(
            y=0.5,
            color='red',
            linestyle='--',
            alpha=0.5,
            label='50% Schwelle'
        )
        ax3.legend(loc='upper right', fontsize=9)
    
    # 4. Statistik-Zusammenfassung
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    improvement = (
        sa_stats['initial_distance'] - sa_stats['final_distance']
    )
    improvement_pct = (improvement / sa_stats['initial_distance']) * 100
    
    stats_text = [
        "═" * 50,
        "SIMULATED ANNEALING - STATISTIKEN",
        "═" * 50,
        "",
        "PERFORMANCE:",
        f"  • Iterationen:        {sa_stats['iterations']:>12,}",
        f"  • Laufzeit:           {sa_stats['runtime']:>12.3f} s",
        f"  • Iter/Sekunde:       "
        f"{sa_stats['iterations']/sa_stats['runtime']:>12,.0f}",
        "",
        "AKZEPTANZ:",
        f"  • Akzeptierte Züge:   {sa_stats['accepted_moves']:>12,}",
        f"  • Verbesserungen:     {sa_stats['improvements']:>12,}",
        f"  • Akzeptanzrate:      {sa_stats['acceptance_rate']:>11.1%}",
        "",
        "DISTANZ:",
        f"  • Initiale Distanz:   {sa_stats['initial_distance']:>12.2f} km",
        f"  • Finale Distanz:     {sa_stats['final_distance']:>12.2f} km",
        f"  • Verbesserung:       {improvement:>12.2f} km",
        f"  • Verbesserung:       {improvement_pct:>11.1f} %",
        "",
        "═" * 50,
    ]
    
    ax4.text(
        0.05, 0.95,
        '\n'.join(stats_text),
        fontsize=10,
        verticalalignment='top',
        family='monospace',
        bbox=dict(
            boxstyle='round',
            facecolor='lightyellow',
            edgecolor='gray',
            alpha=0.8,
            pad=1
        )
    )
    
    plt.suptitle(
        'Simulated Annealing - Konvergenz-Analyse',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Konvergenz-Plot: {filename}")
    plt.close()


def create_comparison_dashboard(
    sa_route: List[str],
    sa_distance: float,
    sa_stats: Dict,
    exact_route: List[str],
    exact_distance: float,
    exact_stats: Dict,
    filename: str = "comparison_dashboard.png"
) -> None:
    """
    Erstellt Vergleichs-Dashboard zwischen SA und exakter Lösung.
    
    Args:
        sa_route: SA-Route
        sa_distance: SA-Distanz
        sa_stats: SA-Statistiken
        exact_route: Optimale Route
        exact_distance: Optimale Distanz
        exact_stats: Statistiken der exakten Lösung
        filename: Ausgabedatei
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Farben
    sa_color = '#3498db'  # Blau
    exact_color = '#2ecc71'  # Grün
    
    # 1. Distanz-Vergleich (Balkendiagramm)
    ax1 = fig.add_subplot(gs[0, 0])
    methods = ['Simulated\nAnnealing', 'Exakte\nLösung']
    distances = [sa_distance, exact_distance]
    colors = [sa_color, exact_color]
    
    bars = ax1.bar(methods, distances, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Distanz (km)', fontsize=11, fontweight='bold')
    ax1.set_title('Distanz-Vergleich', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Werte auf Balken
    for bar, dist in zip(bars, distances):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{dist:.2f} km',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # 2. Laufzeit-Vergleich (Balkendiagramm, log-scale)
    ax2 = fig.add_subplot(gs[0, 1])
    runtimes = [sa_stats['runtime'], exact_stats['runtime']]
    
    bars = ax2.bar(methods, runtimes, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Laufzeit (s)', fontsize=11, fontweight='bold')
    ax2.set_title('Laufzeit-Vergleich', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3)
    
    # Werte auf Balken
    for bar, runtime in zip(bars, runtimes):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{runtime:.3f}s',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # 3. Qualitäts-Gauge
    ax3 = fig.add_subplot(gs[0, 2])
    quality = (exact_distance / sa_distance) * 100
    
    # Kreisdiagramm als Gauge
    colors_gauge = ['green' if quality >= 99 else 'orange', 'lightgray']
    sizes = [quality, 100 - quality]
    
    wedges, texts = ax3.pie(
        sizes,
        colors=colors_gauge,
        startangle=90,
        counterclock=False
    )
    
    # Text in Mitte
    ax3.text(
        0, 0,
        f'{quality:.2f}%',
        ha='center',
        va='center',
        fontsize=24,
        fontweight='bold'
    )
    ax3.text(
        0, -0.3,
        'vom Optimum',
        ha='center',
        va='center',
        fontsize=12
    )
    ax3.set_title('SA Qualität', fontsize=13, fontweight='bold')
    
    # 4. Detaillierte Statistiken (Text)
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    
    dist_diff = sa_distance - exact_distance
    time_speedup = exact_stats['runtime'] / sa_stats['runtime']
    
    stats_table = [
        "═" * 130,
        f"{'ALGORITHMUS-VERGLEICH':<130}",
        "═" * 130,
        f"{'Metrik':<40} {'Simulated Annealing':>25} {'Exakte Lösung':>25} "
        f"{'Differenz':>25}",
        "─" * 130,
        f"{'Distanz (km)':<40} {sa_distance:>25.2f} {exact_distance:>25.2f} "
        f"{dist_diff:>+25.2f}",
        f"{'Laufzeit (s)':<40} {sa_stats['runtime']:>25.3f} "
        f"{exact_stats['runtime']:>25.3f} "
        f"{sa_stats['runtime'] - exact_stats['runtime']:>+25.3f}",
        f"{'Speedup':<40} {'':<25} {'':<25} "
        f"{'SA ' + f'{time_speedup:.1f}x schneller':>25}",
        f"{'Iterationen/Permutationen':<40} {sa_stats['iterations']:>25,} "
        f"{exact_stats.get('iterations', 'N/A'):>25} {'':<25}",
        "─" * 130,
        f"{'QUALITÄTSMETRIKEN':<130}",
        "─" * 130,
        f"{'Optimaler Abstand':<40} {dist_diff:>25.2f} km "
        f"({(dist_diff/exact_distance)*100:>6.2f}%)",
        f"{'Qualität':<40} {quality:>25.2f}% vom Optimum",
        f"{'SA Verbesserung':<40} "
        f"{sa_stats['initial_distance'] - sa_distance:>25.2f} km "
        f"({((sa_stats['initial_distance'] - sa_distance)/sa_stats['initial_distance']*100):>6.2f}%)",
        "═" * 130,
    ]
    
    ax4.text(
        0.5, 0.5,
        '\n'.join(stats_table),
        ha='center',
        va='center',
        fontsize=9,
        family='monospace',
        bbox=dict(
            boxstyle='round',
            facecolor='lightyellow',
            edgecolor='gray',
            alpha=0.8
        )
    )
    
    # 5. SA Konvergenz
    ax5 = fig.add_subplot(gs[2, :2])
    if sa_stats['best_distance_history']:
        iterations = np.arange(len(sa_stats['best_distance_history'])) * 1000
        ax5.plot(
            iterations,
            sa_stats['best_distance_history'],
            color=sa_color,
            linewidth=2,
            label='SA Beste Lösung'
        )
        ax5.axhline(
            y=exact_distance,
            color=exact_color,
            linestyle='--',
            linewidth=2,
            label='Optimale Lösung'
        )
        ax5.fill_between(
            iterations,
            sa_stats['best_distance_history'],
            exact_distance,
            alpha=0.2,
            color='red'
        )
        ax5.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Distanz (km)', fontsize=11, fontweight='bold')
        ax5.set_title(
            'SA Konvergenz zum Optimum',
            fontsize=13,
            fontweight='bold'
        )
        ax5.legend(loc='upper right', fontsize=10)
        ax5.grid(True, alpha=0.3)
    
    # 6. Route-Übereinstimmung
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    # Prüfe Routen-Gleichheit
    routes_equal = (sa_route == exact_route)
    
    route_comparison = [
        "ROUTEN-VERGLEICH",
        "═" * 30,
        "",
        f"SA Route:",
        f"  {' → '.join(sa_route[:3])}...",
        "",
        f"Optimale Route:",
        f"  {' → '.join(exact_route[:3])}...",
        "",
        "─" * 30,
        f"Identisch: {'✓ JA' if routes_equal else '✗ NEIN'}",
        "═" * 30,
    ]
    
    ax6.text(
        0.5, 0.5,
        '\n'.join(route_comparison),
        ha='center',
        va='center',
        fontsize=10,
        family='monospace',
        bbox=dict(
            boxstyle='round',
            facecolor='lightblue' if routes_equal else 'lightyellow',
            edgecolor='gray',
            alpha=0.8
        )
    )
    
    plt.suptitle(
        'TSP Deutschland: Simulated Annealing vs. Exakte Lösung',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Vergleichs-Dashboard: {filename}")
    plt.close()