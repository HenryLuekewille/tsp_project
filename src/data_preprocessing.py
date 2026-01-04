# src/data_preprocessing.py
import pandas as pd
import numpy as np
import os
from math import radians, cos, sin, asin, sqrt

class DestatisfDataProcessor:
    def __init__(self):
        self.base_path = '/Users/henrylukewille/Desktop/tsp_project'
        self.raw_file = os.path.join(
            self.base_path,
            'data/raw/AuszugGV3QAktuell.xlsx'
        )
        self.processed_dir = os.path.join(self.base_path, 'data/processed')
        self.test_dir = os.path.join(self.base_path, 'data/test')
        
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
    
    def load_raw_data_simple(self) -> pd.DataFrame:
        """Lade Destatis-Daten (simple method)"""
        print("📂 Lade Destatis-Daten...")
        
        df = pd.read_excel(
            self.raw_file,
            sheet_name='Onlineprodukt_Gemeinden30092025',
            skiprows=5,
            header=None,
            dtype=str
        )
        
        column_names = [
            'satzart', 'textkennzeichen', 'ars_land', 'ars_rb', 'ars_kreis',
            'ars_vb', 'ars_gem', 'gemeindename', 'flaeche', 'bev_insgesamt',
            'bev_maennlich', 'bev_weiblich', 'bev_dichte', 'postleitzahl',
            'laengengrad', 'breitengrad', 'reisegebiet_schluessel',
            'reisegebiet_bezeichnung', 'verstaedterung_schluessel',
            'verstaedterung_bezeichnung'
        ]
        
        df.columns = column_names
        df = df[df['gemeindename'].notna()].copy()
        df = df[df['gemeindename'].str.len() > 2].copy()
        
        print(f"✅ {len(df)} Zeilen geladen")
        return df
    
    def extract_cities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Städte mit Koordinaten extrahieren"""
        print("\n🏙️  Extrahiere Städtedaten...")
        
        cities_df = pd.DataFrame({
            'name': df['gemeindename'],
            'population': df['bev_insgesamt'],
            'longitude': df['laengengrad'],
            'latitude': df['breitengrad'],
            'flaeche': df['flaeche'],
            'region': df['reisegebiet_bezeichnung']
        })
        
        cities_df = cities_df.dropna(
            subset=['name', 'latitude', 'longitude', 'population']
        )
        
        print("  🔄 Konvertiere Datentypen...")
        cities_df['latitude'] = pd.to_numeric(
            cities_df['latitude'].str.replace(',', '.'), errors='coerce'
        )
        cities_df['longitude'] = pd.to_numeric(
            cities_df['longitude'].str.replace(',', '.'), errors='coerce'
        )
        cities_df['population'] = pd.to_numeric(
            cities_df['population'], errors='coerce'
        )
        cities_df['flaeche'] = pd.to_numeric(
            cities_df['flaeche'], errors='coerce'
        )
        
        cities_df = cities_df.dropna(
            subset=['latitude', 'longitude', 'population']
        )
        
        print("  ✓ Plausibilitäts-Check...")
        initial_count = len(cities_df)
        cities_df = cities_df[
            (cities_df['latitude'] >= 47) & (cities_df['latitude'] <= 56) &
            (cities_df['longitude'] >= 5) & (cities_df['longitude'] <= 16) &
            (cities_df['population'] > 0)
        ]
        
        removed = initial_count - len(cities_df)
        if removed > 0:
            print(f"    ⚠️  {removed} ungültige Einträge entfernt")
        
        cities_df = cities_df.sort_values(
            'population', ascending=False
        ).reset_index(drop=True)
        
        print(f"✅ {len(cities_df)} gültige Städte extrahiert")
        return cities_df
    
    def compute_distance_matrix(self, cities_df: pd.DataFrame) -> np.ndarray:
        """Distanzmatrix mit Haversine-Formel berechnen"""
        n = len(cities_df)
        print(f"\n🧮 Berechne {n}x{n} Distanzmatrix...")
        
        distance_matrix = np.zeros((n, n))
        coords = cities_df[['latitude', 'longitude']].values
        steps = max(1, n // 10)
        
        for i in range(n):
            if i % steps == 0:
                print(f"  Fortschritt: {(i/n)*100:.0f}%")
            
            for j in range(i+1, n):
                lat1, lon1 = radians(coords[i,0]), radians(coords[i,1])
                lat2, lon2 = radians(coords[j,0]), radians(coords[j,1])
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                dist = 6371 * c
                
                distance_matrix[i,j] = distance_matrix[j,i] = dist
        
        print("  Fortschritt: 100%")
        print("✅ Distanzmatrix berechnet")
        return distance_matrix
    
    def create_test_datasets(self, cities_df: pd.DataFrame):
        """Verschiedene Test-Datensätze erstellen"""
        print("\n📦 Erstelle Test-Datensätze...")
        
        # Landeshauptstädte
        hauptstaedte_names = [
            'Berlin, Stadt', 'Hamburg, Freie und Hansestadt',
            'München, Landeshauptstadt', 'Stuttgart, Landeshauptstadt',
            'Düsseldorf, Stadt', 'Hannover, Landeshauptstadt',
            'Bremen, Stadt', 'Dresden, Stadt',
            'Wiesbaden, Landeshauptstadt', 'Potsdam, Stadt',
            'Schwerin, Landeshauptstadt', 'Kiel, Landeshauptstadt',
            'Magdeburg, Landeshauptstadt', 'Mainz, Stadt',
            'Erfurt, Stadt', 'Saarbrücken, Landeshauptstadt'
        ]
        
        hauptstaedte = cities_df[
            cities_df['name'].isin(hauptstaedte_names)
        ].copy()
        
        if len(hauptstaedte) >= 10:
            hauptstaedte.to_csv(
                os.path.join(self.test_dir, 'landeshauptstaedte.csv'),
                index=False, encoding='utf-8'
            )
            print(f"  ✓ Landeshauptstädte: {len(hauptstaedte)} Städte")
        
        # Top 8, 10, 20, 50 - ANGEPASST MIT TOP 10
        for n in [8, 10, 20, 50]:
            subset = cities_df.head(n).copy()
            subset.to_csv(
                os.path.join(self.test_dir, f'top_{n}_staedte.csv'),
                index=False, encoding='utf-8'
            )
            print(f"  ✓ Top {n}: {len(subset)} Städte")
        
        # Städte nach Größe
        for pop_limit in [100000, 50000]:
            subset = cities_df[cities_df['population'] > pop_limit].copy()
            subset.to_csv(
                os.path.join(self.test_dir, f'cities_{pop_limit//1000}k_plus.csv'),
                index=False, encoding='utf-8'
            )
            print(f"  ✓ Städte >{pop_limit//1000}k: {len(subset)} Städte")
    
    def process_all(self):
        """Kompletter Workflow"""
        print("=" * 70)
        print("🚀 DESTATIS-DATEN VERARBEITUNG")
        print("=" * 70)
        
        df = self.load_raw_data_simple()
        cities_df = self.extract_cities(df)
        
        output_file = os.path.join(
            self.processed_dir, 'staedte_complete.csv'
        )
        cities_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n💾 Gespeichert: {output_file}")
        
        self.create_test_datasets(cities_df)
        
        print("\n🧮 Berechne Distanzmatrizen für Test-Sets...")
        
        # ANGEPASST: top_10 hinzugefügt
        test_files = [
            'top_8_staedte.csv',
            'top_10_staedte.csv',
            'top_20_staedte.csv',
            'landeshauptstaedte.csv'
        ]
        
        for test_file in test_files:
            csv_path = os.path.join(self.test_dir, test_file)
            
            # Prüfe ob Datei existiert
            if not os.path.exists(csv_path):
                print(f"  ⚠️  Überspringe {test_file} (nicht gefunden)")
                continue
            
            test_df = pd.read_csv(csv_path)
            dist_matrix = self.compute_distance_matrix(test_df)
            
            matrix_file = test_file.replace('.csv', '_distances.npy')
            np.save(os.path.join(self.test_dir, matrix_file), dist_matrix)
            print(f"  ✓ {matrix_file}")
        
        print("\n" + "=" * 70)
        print("✅ VERARBEITUNG ABGESCHLOSSEN!")
        print("=" * 70)
        
        return cities_df

def main():
    processor = DestatisfDataProcessor()
    cities_df = processor.process_all()
    
    print("\n🏙️  Top 15 größte Städte:")
    top_15 = cities_df.head(15)[['name', 'population']]
    for idx, row in top_15.iterrows():
        print(f"  {idx+1:2d}. {row['name']:<40} "
              f"{int(row['population']):>10,} Einwohner")

if __name__ == "__main__":
    main()