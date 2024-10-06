import pandas as pd
import numpy as np
from multiprocessing import Pool
import pyarrow as pa
import pyarrow.parquet as pq
import gc
from tqdm import tqdm  # Añade esta línea

def haversine_adjusted(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = np.abs(lon2 - lon1)
    dlon = np.where(dlon > np.pi, 2 * np.pi - dlon, dlon)
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    a = np.clip(a, 0, 1) 
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c
    return distance

ruta_base = ''
ruta_archivo = ruta_base + 'specific_time.parquet'
df = pd.read_parquet(ruta_archivo)
specific_time_ds = df[df['time'] == "2014-01-01 00:00:00"].reset_index(drop=True)
del df
gc.collect()

def calculate_distances_for_point(point_idx):
    point_lat, point_lon = specific_time_ds.loc[point_idx, ['LATITUD', 'LONGITUD']]
    distances = specific_time_ds.apply(lambda row: haversine_adjusted(point_lat, point_lon, row['LATITUD'], row['LONGITUD']) if row.name != point_idx else np.inf, axis=1)
    closest_indices = distances.nsmallest(15).index[1:]
    closest_coords = specific_time_ds.loc[closest_indices, ['LATITUD', 'LONGITUD']].values.tolist()
    closest_distances = distances.loc[closest_indices].tolist()
    
    return {'LATITUD': point_lat, 'LONGITUD': point_lon, 'Closest_Coords': closest_coords, 'Distances': closest_distances}

if __name__ == '__main__':
    with Pool(22) as pool:  # Aquí recomiendo cambiar 1 por el número de procesadores que desees utilizar, como 22 que mencionabas.
        num_points = len(specific_time_ds)
        results = []
        for result in tqdm(pool.imap(calculate_distances_for_point, range(num_points)), total=num_points, desc="Calculando distancias"):
            results.append(result)

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df['Closest_Coords'] = results_df['Closest_Coords'].apply(lambda x: x if isinstance(x, list) else [])
        results_df['Distances'] = results_df['Distances'].apply(lambda x: x if isinstance(x, list) else [])

        table = pa.Table.from_pandas(results_df, preserve_index=False)
        pq.write_table(table, 'edges.parquet')
        print("Cálculo de distancias completado y guardado.")
    else:
        print("No se generaron resultados para procesar.")
