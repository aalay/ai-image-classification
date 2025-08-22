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

