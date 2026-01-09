import pandas as pd
import numpy as np
import ollama  # pip install ollama

# --- 1. CONFIGURATION ---
MODEL_NAME = "llama3:latest"  # Or "llama3:latest"
FILE_PATH = r"/home/tom_mscds24/data/Toyota_Final_Merged_Dataset.csv"  # Update this path if needed
OUTPUT_PATH = "Toyota_Informer_Sentiment_Ollama.csv"

# --- 2. SCORING FUNCTION ---
def get_daily_sentiment_ollama(headline_text):

    """
    Uses local Ollama (Llama 3) to score headlines.
    Returns average score (0=Neg, 1=Neu, 2=Pos).
    """
    # A. Handle "No News" -> Neutral (Score 1)
    if str(headline_text).strip() == "No significant news reported" or pd.isna(headline_text):
        return 1.0

    # B. Split into individual headlines (Paper validates on sentences)
    headlines = str(headline_text).split(' | ')
    
    daily_scores = []
    
    for single_headline in headlines:
        # Prompt strictly from Paper [Figure 1]
        prompt_text = (
            f'classify the sentiment of this piece of news headline: "{single_headline}". '
            'Sentiment is "positive", "negative" or "neutral". '
            'Return only the sentiment of the news headline.'
        )

        try:
            # Call Ollama API
            response = ollama.chat(model=MODEL_NAME, messages=[
                {'role': 'user', 'content': prompt_text},
            ])
            
            result = response['message']['content'].strip().lower()
            
            # Map to 0-1-2 Scale
            if "positive" in result:
                daily_scores.append(2)
            elif "negative" in result:
                daily_scores.append(0)
            else:
                daily_scores.append(1) # Neutral (Default for 'neutral' or unclear)
                
        except Exception as e:
            print(f"?? Error on headline: {single_headline[:20]}... -> {e}")
            daily_scores.append(1) # Fallback to Neutral

    # C. Calculate Average for the Day
    if len(daily_scores) > 0:
        return np.mean(daily_scores)
    else:
        return 1.0

# --- 3. EXECUTION LOOP ---
if __name__ == "__main__":
    print(f"?? Loading data from {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH)
    
    print(f"?? Starting Analysis using Local {MODEL_NAME}...")
    final_scores = []

    # Iterate with progress tracking
    for index, row in df.iterrows():
        score = get_daily_sentiment_ollama(row['news_headlines'])
        final_scores.append(score)
        
        if index % 10 == 0:
            print(f"Row {index}/{len(df)} | Score: {score:.2f} | Headlines: {len(str(row['news_headlines']).split('|'))}")

    # --- 4. SAVE RESULTS ---
    df['sentiment_score'] = final_scores
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"? Done! Saved to {OUTPUT_PATH}")
    print(df[['Date', 'sentiment_score']].head())