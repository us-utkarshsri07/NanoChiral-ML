import pandas as pd
import os

def load_real_data():
    base_path = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_path, 'data', 'raw', 'carbon_nanotubes.csv')
    
    df = pd.read_csv(file_path, sep=None, engine='python', dtype=str)
    
   
    for col in df.columns:
        df[col] = df[col].str.replace(',', '.').astype(float)
    new_names = {
        df.columns[0]: 'chiral_n',
        df.columns[1]: 'chiral_m',
        df.columns[2]: 'u_coord',
        df.columns[3]: 'v_coord',
        df.columns[4]: 'initial_w',
        df.columns[7]: 'target_w'
    }
    df = df.rename(columns=new_names)
    
    required_cols = ['chiral_n', 'chiral_m', 'u_coord', 'v_coord', 'initial_w', 'target_w']
    df = df[required_cols]
            
    return df.dropna()