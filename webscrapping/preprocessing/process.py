import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary resources
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Step 1: Read Input Text File
input_file = "../numpy2d2.txt"
output_file = "../numpyprocessed.txt"

with open(input_file, "r", encoding="utf-8") as file:
    raw_data = file.readlines()

# Step 2: Data Cleaning
cleaned_data = []
for text in raw_data:
    text = text.strip()  # Remove leading/trailing whitespace
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags if any
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = text.encode("utf-8", "ignore").decode("utf-8")  # Handle encoding issues
    cleaned_data.append(text)

# Step 3: Handle Missing Data (Replace empty lines with "MISSING_DATA")
cleaned_data = [text if text else "MISSING_DATA" for text in cleaned_data]

# Step 4: Text Normalization
normalized_data = []
for text in cleaned_data:
    words = text.lower().split()  # Convert to lowercase and split words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization & Stopword Removal
    normalized_data.append(" ".join(words))  # Join words back into sentences

# Step 5: Remove Duplicates
df = pd.DataFrame({"text": normalized_data})  # Convert to DataFrame
df.drop_duplicates(inplace=True)  # Remove duplicate rows
final_data = df["text"].tolist()  # Convert back to list

# Step 6: Save the Cleaned Data to a New File
with open(output_file, "w", encoding="utf-8") as file:
    for line in final_data:
        file.write(line + "\n")

print(f"Preprocessing complete! Cleaned data saved to '{output_file}'")