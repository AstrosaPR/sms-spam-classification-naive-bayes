import pickle
import string
import nltk
from tkinter import messagebox as msg
from sklearn.pipeline import Pipeline
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords




# pip install pyinstaller
# pip install auto-py-to-exe

def text_process(mess: str):
    import string
    # mess = mess.lower()
    nopuct = [char for char in mess if char not in string.punctuation]
    nopuct = ''.join(nopuct)
    return [word for word in nopuct.split() if word not in stopwords.words('english')]

with open("spam_detector.pkl", "rb") as model_file:
    model = pickle.load(model_file)




import tkinter as tk
from tkinter import messagebox
from tkinter import *
from tkinter import ttk
from nltk.corpus import stopwords

model_form = Tk()
model_form.title("Spam Detector")
model_form.resizable(0, 0)
model_form.geometry("500x500")
model_form.configure(bg="white")
model_form.iconbitmap("icon/spam_detector_icon.ico")

def predict():
    message = message_input.get(1.0, tk.END).strip()

    if not message:
        messagebox.showwarning("Warning", "Please enter a message.")
        return

    prediction = model.predict([message])
    probability = model.predict_proba([message])

    ham_prob = probability[0][0]
    spam_prob = probability[0][1]

    ham_bar["value"] = ham_prob * 100
    spam_bar["value"] = spam_prob * 100

    ham_percent.configure(text=f"{int(ham_prob * 100)}%")
    spam_percent.configure(text=f"{int(spam_prob * 100)}%")

lbl_message_text = Label(model_form, text="Enter your message: ", bg="white", font=("Arial", 12))
lbl_message_text.grid(column=0, row=0, padx=20, pady=(20, 5), sticky='w')

message_input = Text(model_form, height=5,  width=50, bg="white", font=("Arial", 11))
message_input.grid(column=0, row=1, sticky='w', padx=20, pady=20)

btn_predict = Button(model_form, text="Predict Spam", command=predict, font=('Arial', 12), width=20, height=1)
btn_predict.grid(column=0, row=2, pady=15)

ham_label = tk.Label(model_form, text="Ham: ", bg="white", font=("Arial", 12), fg="green")
ham_label.grid(column=0, row=3, sticky='w', padx=20, pady=(10, 0))

ham_bar = ttk.Progressbar(model_form, length=360, mode="determinate")
ham_bar.grid(column=0, row=4, pady=15, padx=20, sticky='w')

ham_percent = Label(model_form, text='0%', bg="white", font=("Arial", 12))
ham_percent.grid(column=0, row=4, pady=15, padx=20, sticky='e')

spam_label = tk.Label(model_form, text="Spam: ", bg="white", font=("Arial", 12), fg="red")
spam_label.grid(column=0, row=5, sticky='w', padx=20, pady=(10, 0))

spam_bar = ttk.Progressbar(model_form, length=360, mode="determinate")
spam_bar.grid(column=0, row=6, pady=15, padx=20, sticky='w')

spam_percent = Label(model_form, text='0%', bg="white", font=("Arial", 12))
spam_percent.grid(column=0, row=6, pady=15, padx=20, sticky='e')

model_form.mainloop()
