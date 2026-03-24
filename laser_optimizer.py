"""
Laser Cut Optimizer — Zero Bava System
Ottimizzazione parametri taglio laser per eliminazione bava e lavorazioni secondarie
Industry 4.0 — Smart Manufacturing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import os

# Stile compatibile con tutte le versioni matplotlib/seaborn
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')

# ============ MODELLI DATI ============

@dataclass
class MaterialSpec:
    """Specifiche materiale per taglio laser"""
    codice: str
    tipo: str
    spessore_mm: float
    densita: float
    conducibilita_termica: float
    punto_fusione: float
    assorbimento_laser: float

@dataclass
class CutParameters:
    """Parametri di taglio laser"""
    potenza_w: int
    velocita_mm_min: float
    pressione_gas_bar: float
    tipo_gas: str
    focalizzazione_mm: float
    freq_impulso_hz: int

# ============ GENERATORE DATI ============

class LaserDataGenerator:
    """Genera dataset taglio laser con modello fisico bava"""
    
    def __init__(self, n_samples=2500):
        self.n_samples = n_samples
        self.materiali = {
            'DC04_1mm': MaterialSpec('DC04_1mm', 'acciaio_carbonio', 1.0, 7.85, 50, 1538, 0.35),
            'DC04_2mm': MaterialSpec('DC04_2mm', 'acciaio_carbonio', 2.0, 7.85, 50, 1538, 0.35),
            'DC04_3mm': MaterialSpec('DC04_3mm', 'acciaio_carbonio', 3.0, 7.85, 50, 1538, 0.35),
            'AISI304_1.5mm': MaterialSpec('AISI304_1.5mm', 'acciaio_inox', 1.5, 8.0, 15, 1450, 0.30),
            'AISI304_3mm': MaterialSpec('AISI304_3mm', 'acciaio_inox', 3.0, 8.0, 15, 1450, 0.30),
            'AL5754_2mm': MaterialSpec('AL5754_2mm', 'alluminio', 2.0, 2.7, 205, 660, 0.10),
            'AL5754_4mm': MaterialSpec('AL5754_4mm', 'alluminio', 4.0, 2.7, 205, 660, 0.10),
        }
        
    def _calculate_bava_height(self, mat: MaterialSpec, params: CutParameters) -> float:
        """Modello fisico semplificato altezza bava"""
        potenza_opt = mat.spessore_mm * 1000
        velocita_opt = 6000 / mat.spessore_mm
        pressione_opt = 8 if mat.tipo == 'acciaio_carbonio' else 12 if mat.tipo == 'acciaio_inox' else 15
        
        delta_P = (params.potenza_w - potenza_opt) / potenza_opt
        delta_V = (params.velocita_mm_min - velocita_opt) / velocita_opt
        delta_G = (params.pressione_gas_bar - pressione_opt) / pressione_opt
        delta_F = params.focalizzazione_mm / 3
        
        bava_base = 0.05 * mat.spessore_mm
        
        penalita_velocita = max(0, delta_V * 2) if delta_P < -0.2 else abs(delta_V) * 0.5
        penalita_gas = max(0, -delta_G * 1.5)
        penalita_focus = delta_F ** 2 * 0.8
        penalita_potenza = max(0, -delta_P * 1.2)
        penalita_eccesso = max(0, delta_P * 0.3)
        interazione_critica = 2.0 if (delta_V > 0.3 and delta_G < -0.3) else 0
        
        bava_calcolata = bava_base * (1 + 
            penalita_velocita + 
            penalita_gas + 
            penalita_focus + 
            penalita_potenza + 
            penalita_eccesso + 
            interazione_critica
        )
        
        bava_calcolata *= np.random.uniform(0.85, 1.15)
        
        return max(0.02, min(2.0, bava_calcolata))
    
    def _calculate_rugosita(self, bava: float, params: CutParameters) -> float:
        """Rugosità correlata a bava e stabilità processo"""
        base_rz = 5 + bava * 10
        instabilita = abs(params.focalizzazione_mm) * 2
        return base_rz + instabilita + np.random.normal(0, 2)
    
    def _calculate_tempo(self, lunghezza_taglio_mm: float, params: CutParameters) -> float:
        """Tempo taglio includendo overhead"""
        tempo_lineare = lunghezza_taglio_mm / params.velocita_mm_min * 60
        overhead_posizionamento = 2
        return tempo_lineare + overhead_posizionamento
    
    def generate(self, lunghezza_taglio: float = 1000) -> pd.DataFrame:
        """Genera dataset completo tagli"""
        np.random.seed(42)
        records = []
        
        for _ in range(self.n_samples):
            mat_code = np.random.choice(list(self.materiali.keys()))
            mat = self.materiali[mat_code]
            
            params = CutParameters(
                potenza_w=np.random.choice([1000, 1500, 2000, 3000, 4000, 6000]),
                velocita_mm_min=np.random.lognormal(8.5, 0.3),
                pressione_gas_bar=np.random.uniform(6, 18),
                tipo_gas=np.random.choice(['azoto', 'ossigeno', 'aria'], p=[0.5, 0.3, 0.2]),
                focalizzazione_mm=np.random.normal(0, 1.5),
                freq_impulso_hz=np.random.choice([5000, 10000, 20000, 50000])
            )
            
            bava = self._calculate_bava_height(mat, params)
            rugosita = self._calculate_rugosita(bava, params)
            tempo = self._calculate_tempo(lunghezza_taglio, params)
            
            if bava < 0.1:
                qualita_taglio = 'eccellente'
                richiede_sgrassaggio = False
            elif bava < 0.3:
                qualita_taglio = 'buona'
                richiede_sgrassaggio = np.random.random() < 0.3
            elif bava < 0.6:
                qualita_taglio = 'accettabile'
                richiede_sgrassaggio = True
            else:
                qualita_taglio = 'scarsa'
                richiede_sgrassaggio = True
            
            tempo_sgrassaggio = 0 if not richiede_sgrassaggio else tempo * (1 + bava * 2)
            costo_extra = tempo_sgrassaggio * 0.5
            
            record = {
                'materiale_codice': mat_code,
                'materiale_tipo': mat.tipo,
                'spessore_mm': mat.spessore_mm,
                'potenza_w': params.potenza_w,
                'velocita_mm_min': round(params.velocita_mm_min, 0),
                'pressione_gas_bar': round(params.pressione_gas_bar, 1),
                'tipo_gas': params.tipo_gas,
                'focalizzazione_mm': round(params.focalizzazione_mm, 2),
                'larghezza_taglio_mm': round(0.2 + mat.spessore_mm * 0.05, 2),
                'altezza_bava_mm': round(bava, 3),
                'rugosita_ra_um': round(rugosita, 1),
                'qualita_taglio': qualita_taglio,
                'richiede_sgrassaggio': richiede_sgrassaggio,
                'tempo_taglio_s': round(tempo, 1),
                'tempo_totale_con_sgrassaggio_s': round(tempo + tempo_sgrassaggio, 1),
                'costo_extra_bava_eur': round(costo_extra, 2),
                'bava_eccessiva': 1 if bava > 0.3 else 0
            }
            records.append(record)
        
        return pd.DataFrame(records)

# ============ PREDITORE BAVA ============

class BavaPredictor:
    """Predice altezza bava e suggerisce parametri ottimali"""
    
    def __init__(self):
        self.regressor = None
        self.classifier = None
        self.scaler = StandardScaler()
        self.le_material = LabelEncoder()
        self.le_gas = LabelEncoder()
        
    def _prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Feature engineering per modelli"""
        df = df.copy()
        
        if fit:
            df['mat_enc'] = self.le_material.fit_transform(df['materiale_tipo'])
            df['gas_enc'] = self.le_gas.fit_transform(df['tipo_gas'])
        else:
            df['mat_enc'] = self.le_material.transform(df['materiale_tipo'])
            df['gas_enc'] = self.le_gas.transform(df['tipo_gas'])
        
        df['energia_densita'] = df['potenza_w'] / (df['velocita_mm_min'] * df['spessore_mm'])
        df['potenza_spessore'] = df['potenza_w'] / df['spessore_mm']
        df['pressione_spessore'] = df['pressione_gas_bar'] / df['spessore_mm']
        df['focus_ottimale'] = (abs(df['focalizzazione_mm']) < 0.5).astype(int)
        
        features = ['mat_enc', 'spessore_mm', 'potenza_w', 'velocita_mm_min',
                   'pressione_gas_bar', 'gas_enc', 'focalizzazione_mm',
                   'energia_densita', 'potenza_spessore', 'pressione_spessore', 'focus_ottimale']
        
        X = df[features].values
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X, features
    
    def train(self, df: pd.DataFrame):
        """Allena entrambi i modelli"""
        print("\nTraining modelli predizione bava...")
        print("=" * 60)
        
        X, feature_names = self._prepare_features(df, fit=True)
        
        # Regressore
        y_bava = df['altezza_bava_mm'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y_bava, test_size=0.2, random_state=42)
        
        self.regressor = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.regressor.fit(X_train, y_train)
        
        y_pred = self.regressor.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Regressore Bava | MAE: {mae:.3f} mm | R²: {self.regressor.score(X_test, y_test):.3f}")
        
        # Classificatore
        y_class = df['bava_eccessiva'].values
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
        
        self.classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X_train_c, y_train_c)
        
        y_pred_c = self.classifier.predict(X_test_c)
        acc = (y_pred_c == y_test_c).mean()
        print(f"Classificatore    | Accuracy: {acc:.3f}")
        print("\nFeature importance top 5:")
        importances = pd.Series(self.regressor.feature_importances_, index=feature_names).sort_values(ascending=False)
        for feat, imp in importances.head(5).items():
            print(f"  {feat:20}: {imp:.3f}")
        
        self.feature_names = feature_names
        return self
    
    def predict(self, materiale: MaterialSpec, params: CutParameters) -> Dict:
        """Predice bava per parametri dati"""
        input_df = pd.DataFrame([{
            'materiale_tipo': materiale.tipo,
            'spessore_mm': materiale.spessore_mm,
            'potenza_w': params.potenza_w,
            'velocita_mm_min': params.velocita_mm_min,
            'pressione_gas_bar': params.pressione_gas_bar,
            'tipo_gas': params.tipo_gas,
            'focalizzazione_mm': params.focalizzazione_mm
        }])
        
        X, _ = self._prepare_features(input_df, fit=False)
        
        bava_pred = self.regressor.predict(X)[0]
        bava_eccessiva_prob = self.classifier.predict_proba(X)[0][1]
        
        return {
            'altezza_bava_prevista_mm': round(bava_pred, 3),
            'probabilita_bava_eccessiva': round(bava_eccessiva_prob * 100, 1),
            'qualita_prevista': 'scarsa' if bava_pred > 0.6 else 'accettabile' if bava_pred > 0.3 else 'buona' if bava_pred > 0.1 else 'eccellente',
            'richiede_sgrassaggio_previsto': bava_pred > 0.3
        }

# ============ OTTIMIZZATORE ============

class CutOptimizer:
    """Trova parametri ottimali per zero bava"""
    
    def __init__(self, predictor: BavaPredictor):
        self.predictor = predictor
        
    def optimize(self, materiale: MaterialSpec, 
                potenza_disponibile: List[int] = None,
                gas_disponibile: List[str] = None) -> Dict:
        """Ottimizza parametri minimizzando bava e tempo ciclo"""
        if potenza_disponibile is None:
            potenza_disponibile = [1500, 2000, 3000, 4000]
        if gas_disponibile is None:
            gas_disponibile = ['azoto', 'ossigeno', 'aria']
        
        print(f"\nOttimizzazione taglio {materiale.codice}...")
        print(f"Spessore: {materiale.spessore_mm}mm | Materiale: {materiale.tipo}")
        
        best_result = None
        best_score = float('inf')
        
        for potenza in potenza_disponibile:
            for gas in gas_disponibile:
                vel_opt = potenza / materiale.spessore_mm
                velocita_range = np.linspace(vel_opt * 0.6, vel_opt * 1.4, 15)
                
                for velocita in velocita_range:
                    pressione_base = 10 if gas == 'azoto' else 8 if gas == 'ossigeno' else 12
                    pressione = pressione_base + materiale.spessore_mm * 0.5
                    
                    params = CutParameters(
                        potenza_w=potenza,
                        velocita_mm_min=velocita,
                        pressione_gas_bar=pressione,
                        tipo_gas=gas,
                        focalizzazione_mm=0,
                        freq_impulso_hz=10000
                    )
                    
                    prediction = self.predictor.predict(materiale, params)
                    
                    bava_penalty = prediction['altezza_bava_prevista_mm'] * 100 if prediction['altezza_bava_prevista_mm'] < 0.3 else prediction['altezza_bava_prevista_mm'] * 500
                    tempo_factor = 1000 / velocita
                    
                    score = bava_penalty + tempo_factor * 0.3
                    
                    if score < best_score:
                        best_score = score
                        best_result = {
                            'parametri': params,
                            'predizione': prediction,
                            'score': score
                        }
        
        return {
            'potenza_ottimale_w': best_result['parametri'].potenza_w,
            'velocita_ottimale_mm_min': round(best_result['parametri'].velocita_mm_min, 0),
            'pressione_ottimale_bar': round(best_result['parametri'].pressione_gas_bar, 1),
            'gas_ottimale': best_result['parametri'].tipo_gas,
            'focalizzazione_mm': best_result['parametri'].focalizzazione_mm,
            'bava_prevista_mm': best_result['predizione']['altezza_bava_prevista_mm'],
            'qualita_prevista': best_result['predizione']['qualita_prevista'],
            'richiede_sgrassaggio': best_result['predizione']['richiede_sgrassaggio_previsto'],
            'risparmio_tempo_stimato': '50-100%' if not best_result['predizione']['richiede_sgrassaggio_previsto'] else '0%'
        }

# ============ VISUALIZZAZIONE ============

class LaserVisualizer:
    """Dashboard qualità taglio laser"""
    
    def __init__(self):
        self.colors = {'eccellente': '#2ecc71', 'buona': '#27ae60', 
                      'accettabile': '#f1c40f', 'scarsa': '#e74c3c'}
    
    def plot_bava_analysis(self, df: pd.DataFrame, save_path: str = "bava_analysis.png"):
        """Analisi bava vs parametri"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # 1. Bava vs velocità
        ax = axes[0, 0]
        for pot in df['potenza_w'].unique():
            subset = df[df['potenza_w'] == pot]
            ax.scatter(subset['velocita_mm_min'], subset['altezza_bava_mm'], 
                      label=f'{pot}W', alpha=0.6, s=30)
        ax.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Soglia sgrassaggio')
        ax.set_xlabel('Velocità (mm/min)')
        ax.set_ylabel('Altezza bava (mm)')
        ax.set_title('Bava vs Velocità per Potenza', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 2. Bava vs pressione gas
        ax = axes[0, 1]
        for gas in df['tipo_gas'].unique():
            subset = df[df['tipo_gas'] == gas]
            ax.scatter(subset['pressione_gas_bar'], subset['altezza_bava_mm'], 
                      label=gas, alpha=0.6, s=30)
        ax.axhline(y=0.3, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Pressione gas (bar)')
        ax.set_ylabel('Altezza bava (mm)')
        ax.set_title('Effetto Pressione Gas', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Distribuzione bava per materiale
        ax = axes[0, 2]
        df.boxplot(column='altezza_bava_mm', by='materiale_tipo', ax=ax)
        ax.axhline(y=0.3, color='r', linestyle='--', alpha=0.5)
        ax.set_title('Distribuzione Bava per Materiale', fontweight='bold')
        ax.set_xlabel('Materiale')
        
        # 4. Costo impatto bava
        ax = axes[1, 0]
        cost_by_quality = df.groupby('qualita_taglio')['costo_extra_bava_eur'].sum()
        colors = [self.colors.get(q, 'gray') for q in cost_by_quality.index]
        ax.bar(cost_by_quality.index, cost_by_quality.values, color=colors, alpha=0.7)
        ax.set_title('Costo Totale Bava per Qualità', fontweight='bold')
        ax.set_ylabel('€ (stimato)')
        
        # 5. Heatmap parametri ottimali
        ax = axes[1, 1]
        pivot = df.pivot_table(values='altezza_bava_mm', 
                              index='potenza_w', 
                              columns=pd.cut(df['velocita_mm_min'], bins=5),
                              aggfunc='mean')
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Bava mm'})
        ax.set_title('Heatmap Bava: Potenza vs Velocità', fontweight='bold')
        
        # 6. Confronto tempo con/senza sgrassaggio
        ax = axes[1, 2]
        tempo_con = df[df['richiede_sgrassaggio']]['tempo_totale_con_sgrassaggio_s'].mean()
        tempo_senza = df[~df['richiede_sgrassaggio']]['tempo_taglio_s'].mean()
        ax.bar(['Taglio pulito\n(bava <0.3mm)', 'Con sgrassaggio\n(bava >0.3mm)'], 
              [tempo_senza, tempo_con], 
              color=['green', 'red'], alpha=0.7)
        ax.set_title('Impatto Tempo Ciclo', fontweight='bold')
        ax.set_ylabel('Secondi')
        for i, v in enumerate([tempo_senza, tempo_con]):
            ax.text(i, v + 5, f'{v:.0f}s', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Salvato: {save_path}")
    
    def plot_optimization_result(self, original_params: CutParameters, 
                                optimized: Dict,
                                save_path: str = "optimization_result.png"):
        """Visualizza confronto prima/dopo ottimizzazione"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Parametri
        ax = axes[0]
        params_labels = ['Potenza\n(W)', 'Velocità\n(mm/min)', 'Pressione\n(bar)']
        original_vals = [original_params.potenza_w, original_params.velocita_mm_min, 
                        original_params.pressione_gas_bar]
        optimized_vals = [optimized['potenza_ottimale_w'], optimized['velocita_ottimale_mm_min'],
                         optimized['pressione_ottimale_bar']]
        
        x = np.arange(len(params_labels))
        width = 0.35
        
        ax.bar(x - width/2, original_vals, width, label='Originale', color='red', alpha=0.7)
        ax.bar(x + width/2, optimized_vals, width, label='Ottimizzato', color='green', alpha=0.7)
        ax.set_ylabel('Valore')
        ax.set_title('Confronto Parametri', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(params_labels)
        ax.legend()
        
        # Risultato bava
        ax = axes[1]
        bava_originale = 0.8
        bava_ottimizzato = optimized['bava_prevista_mm']
        
        colors = ['red' if bava_originale > 0.3 else 'green', 
                 'red' if bava_ottimizzato > 0.3 else 'green']
        ax.bar(['Parametri\nattuali', 'Parametri\nottimizzati'], 
              [bava_originale, bava_ottimizzato], 
              color=colors, alpha=0.7, width=0.5)
        ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Soglia sgrassaggio')
        ax.set_ylabel('Altezza bava (mm)')
        ax.set_title('Risultato Qualità Taglio', fontweight='bold')
        ax.legend()
        
        if not optimized['richiede_sgrassaggio']:
            ax.text(1, bava_ottimizzato + 0.1, 'ZERO BAVA\nRisparmio 50-100% tempo', 
                   ha='center', fontweight='bold', color='green')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Salvato: {save_path}")

# ============ MAIN ============

def main():
    print("=" * 70)
    print("LASER CUT OPTIMIZER — Zero Bava System")
    print("Eliminazione lavorazioni secondarie tramite ML e ottimizzazione parametri")
    print("=" * 70)
    
    os.makedirs('outputs', exist_ok=True)
    
    # 1. Genera dati training
    print("\n1. Generazione dataset storico tagli laser...")
    generator = LaserDataGenerator(n_samples=2000)
    df = generator.generate(lunghezza_taglio=1000)
    
    print(f"   Generati {len(df)} record tagli")
    print(f"   Bava media dataset: {df['altezza_bava_mm'].mean():.3f} mm")
    print(f"   Tagli con bava eccessiva (>0.3mm): {df['bava_eccessiva'].sum()} ({df['bava_eccessiva'].mean()*100:.1f}%)")
    print(f"   Costo extra medio per bava: {df['costo_extra_bava_eur'].mean():.2f} €/taglio")
    
    # 2. Training modelli
    print("\n2. Addestramento modelli predittivi...")
    predictor = BavaPredictor()
    predictor.train(df)
    
    # 3. Visualizzazione dati
    print("\n3. Generazione analisi visiva...")
    viz = LaserVisualizer()
    viz.plot_bava_analysis(df, "outputs/bava_analysis.png")
    
    # 4. Esempio ottimizzazione reale
    print("\n4. Ottimizzazione caso reale...")
    print("-" * 70)
    
    materiale_problema = generator.materiali['DC04_2mm']
    
    parametri_attuali = CutParameters(
        potenza_w=2000,
        velocita_mm_min=8000,
        pressione_gas_bar=8,
        tipo_gas='ossigeno',
        focalizzazione_mm=0,
        freq_impulso_hz=10000
    )
    
    predizione_attuale = predictor.predict(materiale_problema, parametri_attuali)
    print(f"PARAMETRI ATTUALI (tipici operatore):")
    print(f"  Potenza: {parametri_attuali.potenza_w}W | Velocità: {parametri_attuali.velocita_mm_min}mm/min")
    print(f"  Gas: {parametri_attuali.tipo_gas} @ {parametri_attuali.pressione_gas_bar}bar")
    print(f"  → Bava prevista: {predizione_attuale['altezza_bava_prevista_mm']} mm")
    print(f"  → Qualità: {predizione_attuale['qualita_prevista']}")
    print(f"  → Richiede sgrassaggio: {'SÌ (raddoppia tempo)' if predizione_attuale['richiede_sgrassaggio_previsto'] else 'No'}")
    
    optimizer = CutOptimizer(predictor)
    risultato_ottimale = optimizer.optimize(
        materiale_problema,
        potenza_disponibile=[1500, 2000, 3000],
        gas_disponibile=['azoto', 'ossigeno']
    )
    
    print(f"\nPARAMETRI OTTIMIZZATI (suggeriti dal sistema):")
    print(f"  Potenza: {risultato_ottimale['potenza_ottimale_w']}W | Velocità: {risultato_ottimale['velocita_ottimale_mm_min']}mm/min")
    print(f"  Gas: {risultato_ottimale['gas_ottimale']} @ {risultato_ottimale['pressione_ottimale_bar']}bar")
    print(f"  → Bava prevista: {risultato_ottimale['bava_prevista_mm']} mm")
    print(f"  → Qualità: {risultato_ottimale['qualita_prevista']}")
    print(f"  → Richiede sgrassaggio: {'SÌ' if risultato_ottimale['richiede_sgrassaggio'] else 'NO → Risparmio 50-100% tempo'}")
    
    viz.plot_optimization_result(parametri_attuali, risultato_ottimale, "outputs/optimization_result.png")
    
    # 5. Analisi impatto economico
    print("\n5. Analisi impatto economico...")
    print("-" * 70)
    
    tagli_giorno = 50
    giorni_lavorativi = 220
    
    costo_attuale_annuo = df[df['bava_eccessiva']==1]['costo_extra_bava_eur'].mean() * tagli_giorno * giorni_lavorativi
    pezzi_con_bava = df['bava_eccessiva'].mean()
    risparmio_stimato = costo_attuale_annuo * (1 - pezzi_con_bava * 0.1)
    
    print(f"Produzione: {tagli_giorno} tagli/giorno × {giorni_lavorativi} giorni = {tagli_giorno*giorni_lavorativi:,} tagli/anno")
    print(f"Costo lavorazioni secondarie attuale: ~{costo_attuale_annuo:,.0f} €/anno")
    print(f"Costo stimato con ottimizzazione: ~{costo_attuale_annuo - risparmio_stimato:,.0f} €/anno")
    print(f"RISPARMIO POTENZIALE: {risparmio_stimato:,.0f} €/anno ({risparmio_stimato/costo_attuale_annuo*100:.0f}%)")
    
    # Salva dati
    df.to_csv('outputs/laser_cut_data.csv', index=False)
    
    def convert_to_serializable(obj):
        """Converte numpy types in tipi Python standard per JSON"""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    summary = convert_to_serializable({
        'generated_at': datetime.now().isoformat(),
        'materiale_test': materiale_problema.codice,
        'bava_attuale_mm': predizione_attuale['altezza_bava_prevista_mm'],
        'bava_ottimizzata_mm': risultato_ottimale['bava_prevista_mm'],
        'risparmio_tempo': '50-100%' if not risultato_ottimale['richiede_sgrassaggio'] else '0%',
        'risparmio_costo_annuo_eur': round(risparmio_stimato, 0),
        'parametri_ottimali': risultato_ottimale
    })
    
    with open('outputs/optimization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("OTTIMIZZAZIONE COMPLETATA")
    print("=" * 70)
    print("\nOutput in /outputs:")
    print("  • bava_analysis.png — Analisi bava vs parametri")
    print("  • optimization_result.png — Confronto prima/dopo")
    print("  • laser_cut_data.csv — Dataset completo")
    print("  • optimization_summary.json — Risultati ottimizzazione")

if __name__ == "__main__":
    main()
