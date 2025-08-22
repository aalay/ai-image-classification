import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Class names
class_names = ['airplane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nTest accuracy: {test_acc:.2f}')

# Optional: Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.title('Training and Validation Accuracy')
plt.show()
import nltk
from nltk.chat.util import Chat, reflections

# Sample conversation patterns
pairs = [
    [
        r"hi|hello|hey",
        ["Hello!", "Hi there!", "Hey!"]
    ],
    [
        r"what is your name ?",
        ["I'm a chatbot created by you."]
    ],
    [
        r"how are you ?",
        ["I'm doing well, thank you!", "All good! How about you?"]
    ],
    [
        r"sorry (.*)",
        ["No problem!", "It's okay!", "Don't worry about it."]
    ],
    [
        r"(.*) (location|city) ?",
        ["I'm in the cloud :)"]
    ],
    [
        r"quit",
        ["Bye! Have a great day!"]
    ],
    [
        r"(.*)",
        ["Sorry, I didn't understand that."]
    ]
]

# Create the chatbot
chatbot = Chat(pairs, reflections)

# Start chat
print("Hello! I'm your chatbot. Type 'quit' to exit.")
chatbot.converse()
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer  # You can try LexRankSummarizer or LuhnSummarizer too

# Input text
text = """
Artificial Intelligence (AI) is rapidly transforming industries around the world.
From healthcare and finance to transportation and manufacturing, AI is driving innovation.
It is enabling machines to perform tasks that typically require human intelligence.
These include speech recognition, decision-making, and language translation.
However, with the rise of AI come challenges such as ethical concerns and job displacement.
It's crucial to balance progress with responsibility as we integrate AI into daily life.
"""

# Parse text
parser = PlaintextParser.from_string(text, Tokenizer("english"))

# Summarize
summarizer = LsaSummarizer()
summary = summarizer(parser.document, 2)  # 2 sentences in summary

# Output
print("Summary:")
for sentence in summary:
    print("-", sentence)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
data = {
    'title': ['Inception', 'Interstellar', 'The Matrix', 'The Prestige', 'Memento'],
    'description': [
        'A thief who steals corporate secrets through dream-sharing technology.',
        'A team travels through a wormhole in space in an attempt to ensure humanity’s survival.',
        'A computer hacker learns about the true nature of reality and his role in the war.',
        'Two magicians engage in a battle to create the ultimate illusion.',
        'A man with short-term memory loss attempts to track down his wife’s murderer.'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Convert text to TF-IDF features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# Compute cosine similarity between all movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies
def recommend(title, df=df, similarity=cosine_sim):
    if title not in df['title'].values:
        return "Movie not found."
    
    idx = df[df['title'] == title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    recommended_indices = [i[0] for i in scores[1:4]]  # top 3 similar
    return df['title'].iloc[recommended_indices].tolist()

# Test it
movie = "Inception"
print(f"Movies similar to '{movie}':")
print(recommend(movie))

