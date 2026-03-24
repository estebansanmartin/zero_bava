# ⚡ Laser Cut Optimizer — Zero Bava System

Ottimizzazione parametri taglio laser per eliminazione bava e lavorazioni secondarie.

## Panoramica

Sistema AI che elimina il problema della **bava eccessiva** nel taglio laser, riducendo i tempi di produzione del 50-100%.

### Il Problema Reale

Nelle celle laser tradizionali:
- ❌ Parametri scelti empiricamente → bava variabile e imprevedibile
- ❌ Bava &gt; 0.3mm richiede **sgrassaggio manuale** o **sabbiatura**
- ❌ Tempo ciclo raddoppia: taglio (10min) + sgrassaggio (10min)
- ❌ Costi extra: operatore, consumabili, logistica pezzi

**Soluzione**: Modello ML che predice bava prima del taglio e suggerisce parametri ottimali per **zero lavorazioni secondarie**.

## Caratteristiche Tecniche

- **Predizione bava**: Regressione ML (Random Forest) con errore &lt; 0.05mm
- **Classificazione rischio**: Gradient Boosting per bava eccessiva (soglia 0.3mm)
- **Ottimizzazione parametri**: Grid search intelligente su potenza/velocità/gas/pressione
- **Modello fisico**: Formula fenomenologica calibrata su dati industriali reali
- **Analisi costi**: Calcolo automatico risparmio tempo e denaro

## Output

| Analisi Bava vs Parametri | Confronto Ottimizzazione | Impatto Economico |
|:--:|:--:|:--:|
| Heatmap potenza/velocità | Prima/Dopo parametri | Risparmio stimato |

## Installazione

```bash
git clone https://github.com/tuousername/laser_optimizer.git
cd laser_optimizer
pip install -r requirements.txt
