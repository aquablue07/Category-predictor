import spacy
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Load all saved trained components
def load_components(model_dir="model")
    model_dir = Path(model_dir)    
    nlp = spacy.load(model_dir / "nlp")
    label_encoder = joblib.load(model_dir / "label_encoder.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    
    return nlp, label_encoder, scaler

def prepare_unlabeled_data(df, scaler): # Same as in training
  
    if 'Processed_Text' not in df.columns:
        if 'text' in df.columns:
            df['Processed_Text'] = df['text']
        else:
            raise ValueError("No text column found")
    
    df['Processed_Text'] = df['Processed_Text'].astype(str).str.strip()   #some float values are hidden in some text rows
    
    if 'Manufacturer' in df.columns:
        df['Manufacturer'] = df['Manufacturer'].astype(str).str.strip().fillna('unknown')
    else:
        df['Manufacturer'] = 'unknown'  # Default value
    
    # normalize weights
    if 'Weight_lbs' in df.columns:
        df['weight_norm'] = scaler.transform(df[['Weight_lbs']]).flatten()
    else:
        df['weight_norm'] = 0.0  # Neutral scaled value
    
    return df


def predict_categories(nlp, df, label_encoder, scaler):
  
    results = []
    
    for _, row in df.iterrows():
        doc = nlp.make_doc(row['Processed_Text'])
        doc._.weight = row['weight_norm']
        doc._.manufacturer = row['Manufacturer']
        pred = nlp(doc)
        
        
        pred_category, confidence = max(pred.cats.items(), key=lambda x: x[1]) # Get top prediction
        
        results.append({
            'text': row['Processed_Text'],
            'predicted_category': pred_category,
            'confidence': confidence,
            'manufacturer': row['Manufacturer'],
            'weight': row.get('Weight_lbs', None)
        })
    
    return pd.DataFrame(results)

# Main execution
def main(unlabeled_path="Downloads/final_spacyunlabeled.csv"):
  
  print("Loading trained model...")
  nlp, label_encoder, scaler = load_components()
  
  print("Loading unlabeled data...")
  unlabeled_df = pd.read_csv(unlabeled_path)
  
  print("Preprocessing data...")
  prepared_df = prepare_unlabeled_data(unlabeled_df, scaler)
  
  print("Making predictions...")
  predictions = predict_categories(nlp, prepared_df, label_encoder, scaler)
  
  output_path = "predictions.csv"
  predictions.to_csv(output_path, index=False)
  print(f"Predictions saved to {output_path}")
  
  return predictions
    
    

if __name__ == "__main__":
    predictions = main()
    #print("\nSample predictions:")   #Enable if using jupyter or if you need osme sample preds
    #print(predictions.head())
