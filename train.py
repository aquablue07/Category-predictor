# Classification layer on top of SpaCy
import spacy  
from spacy.tokens import Doc  
from spacy.training import Example 
from spacy.util import minibatch  
import pandas as pd  
import numpy as np 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder, StandardScaler  
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, precision_recall_fscore_support  
from pathlib import Path 
from collections import defaultdict  
import joblib 

def load_data(filepath, sample_frac=1.0):
 
    df = pd.read_csv(filepath) 
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=7)  # Randomly select rows
    return df

def prepare_data(df):
    
    df['Processed_Text'] = df['Processed_Text'].astype(str).str.strip()   # some text data has float in some rows. standardize to str
    df['Manufacturer'] = df['Manufacturer'].astype(str).str.strip().fillna('unknown') 

    le = LabelEncoder()  
    df['category_id'] = le.fit_transform(df['Category'])  

    scaler = StandardScaler()  
    df['weight_norm'] = scaler.fit_transform(df[['Weight_lbs']]).flatten()  # Scale weight values
    return df, le, scaler

def initialize_spacy(classes):      #Initializes a spaCy NLP pipeline for text classification.
    
    nlp = spacy.blank("en") 
    textcat = nlp.add_pipe("textcat", last=True)  # Add a text classification component
    for cat in classes:
        textcat.add_label(cat)

  
    Doc.set_extension("weight", default=0.0, force=True)      
    Doc.set_extension("manufacturer", default="", force=True) 

    return nlp

def example_generator(df, nlp):      #Generates spaCy Example objects to feed to the training pipeline
    
    for _, row in df.iterrows():
        doc = nlp.make_doc(row['Processed_Text'])  
        doc._.weight = row['weight_norm'] 
        doc._.manufacturer = row['Manufacturer']  
        yield Example.from_dict(doc, {"cats": {row['Category']: 1.0}}) 

def train_model(nlp, train_df, val_df, label_encoder, n_iter=10, batch_size=8):  #Train the model
    
    optimizer = nlp.begin_training() 
    best_accuracy = 0.0 
    history = []  
    model_dir = Path("model_temp")  
    model_dir.mkdir(exist_ok=True)  

    for epoch in range(n_iter):
        train_loss = 0.0  
        examples = list(example_generator(train_df, nlp))  
        random.shuffle(examples) 

        for batch in minibatch(examples, size=batch_size):  
            losses = {} 
            nlp.update(batch, losses=losses, sgd=optimizer) 
            train_loss += losses.get('textcat', 0.0)  #aggregate training loss

        train_acc = calculate_accuracy(nlp, train_df)  # Calculate training accuracy
        val_acc = calculate_accuracy(nlp, val_df)  # Calculate validation accuracy

        if val_acc > best_accuracy:  
            best_accuracy = val_acc  
            best_model_path = model_dir / f"best_model"  
            nlp.to_disk(best_model_path)    # keep updating the best model, whenever the validation accuracy gets better
        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'train_acc': train_acc, 'val_acc': val_acc}) 
        print(f"Epoch {epoch+1}/{n_iter}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return spacy.load(best_model_path), history 

def calculate_accuracy(nlp, df):  

    correct = 0 
    for _, row in df.iterrows():
        doc = nlp(row['Processed_Text'])  
        pred = max(doc.cats.items(), key=lambda x: x[1])[0]  
        if pred == row['Category']:  
            correct += 1  
    return correct / len(df)  

def save_components(nlp, label_encoder, scaler, output_dir="model"): #Saves the trained model and preprocessing components.

    output_path = Path(output_dir)  
    output_path.mkdir(exist_ok=True)  
    nlp.to_disk(output_path / "nlp")  # Save the spaCy model
    joblib.dump(label_encoder, output_path / "label_encoder.joblib")  # Save the LabelEncoder
    joblib.dump(scaler, output_path / "scaler.joblib")  # Save the StandardScaler
    print(f"Model saved to {output_path}")



def main(data_path="Downloads/final_spacy.csv"):
   
    df = load_data(data_path, sample_frac=0.8)       # loading  a subset of total data due to memory constrainsts. change it back to 1
    df, label_encoder, scaler = prepare_data(df)  
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=7) 
    nlp = initialize_spacy(label_encoder.classes_)  # Initialize the spaCy model
    trained_nlp, history = train_model(nlp, train_df, val_df, label_encoder, 10)  
    save_components(trained_nlp, label_encoder, scaler)
    return trained_nlp

if __name__ == "__main__":
    nlp = main()  
    
