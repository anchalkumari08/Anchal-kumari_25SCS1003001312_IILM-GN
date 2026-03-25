import os, tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np, re, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

UNKNOWN_FOLDER = "training_unknown"

def load_txt(path):
    try:
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    except:
        return ""

def load_folder(folder):
    valid_texts = []
    skipped = []

    if not os.path.exists(folder):
        return []

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".txt"):
            continue

        path = os.path.join(folder, fname)
        try:
            text = load_txt(path).strip()
        except:
            skipped.append(fname)
            continue

        words_list = words(text)

        # Filters:
        if len(text) < 10:
            skipped.append(fname); continue

        if len(words_list) < 20:
            skipped.append(fname); continue

        # Check if at least one non-stopword exists
        meaningful = [w for w in words_list if w not in {
            "the","and","is","in","it","of","to","a","that","i","you",
            "was","for","on","with","as","he","she","they","we","this"
        }]
        if len(meaningful) == 0:
            skipped.append(fname); continue

        valid_texts.append(text)

    # Show a warning if anything was skipped
    if skipped:
        messagebox.showwarning(
            "Skipped Files",
            "Some files were skipped because they were too short or empty:\n" + "\n".join(skipped)
        )

    return valid_texts

def words(t): return re.findall(r"\b\w+\b", t.lower())
def sents(t): return re.split(r"[.!?]+", t)

def style_vec(t):
    w = words(t)
    if not w: return np.zeros(4)
    wl = len(w)
    return np.array([
        sum(len(x) for x in w)/wl,
        sum(1 for x in t if x in ".,;:!?")/wl,
        sum(len(words(s)) for s in sents(t))/max(len(sents(t)),1),
        sum(1 for x in w if len(x)<4)/wl
    ])

def build_pairs(texts, n=20):
    snippets = []
    for t in texts:
        w = words(t)
        if len(w) < 60: continue
        for _ in range(max(1, n//2)):
            start = random.randint(0, max(0, len(w)-60))
            snippets.append(" ".join(w[start:start+60]))
    same, diff = [], []
    # same-author pairs from same doc (adjacent)
    for i in range(0, len(snippets)-1, 2):
        same.append((snippets[i], snippets[i+1]))
    # different-author pairs by random sampling
    for _ in range(len(same)*2 + 20):
        a, b = random.sample(snippets, 2)
        diff.append((a,b))
    # keep balanced
    m = min(len(same), len(diff))
    return same[:m], diff[:m]

def safe_fit_vectorizers(corpus):
    # try with english stop words, fallback to no stop words if empty vocabulary arises
    try:
        v1 = TfidfVectorizer(stop_words="english", max_features=20000)
        v1.fit(corpus)
        v2 = TfidfVectorizer(use_idf=True, binary=True, stop_words="english")
        v2.fit(corpus)
    except ValueError:
        # fallback: no stop words
        v1 = TfidfVectorizer(stop_words=None, max_features=20000, token_pattern=r"(?u)\b\w+\b")
        v1.fit(corpus)
        v2 = TfidfVectorizer(use_idf=True, binary=True, stop_words=None, token_pattern=r"(?u)\b\w+\b")
        v2.fit(corpus)
    return v1, v2

def train_meta(texts):
    if len(texts) < 3:
        raise ValueError("Need at least 3 unknown-author .txt files (longer texts are better).")
    same, diff = build_pairs(texts)
    if len(same) == 0 or len(diff) == 0:
        raise ValueError("Not enough snippet pairs created from unknown texts. Make unknown files longer / add more files.")
    pairs = same + diff
    labels = [1]*len(same) + [0]*len(diff)
    corpus = [a for a,b in pairs] + [b for a,b in pairs]
    tfidf, idfbin = safe_fit_vectorizers(corpus)
    X = []
    for a,b in pairs:
        v1,v2 = tfidf.transform([a,b])
        i1,i2 = idfbin.transform([a,b])
        s1,s2 = style_vec(a), style_vec(b)
        X.append([
            cosine_similarity(v1,v2)[0][0],
            cosine_similarity(i1,i2)[0][0],
            cosine_similarity([s1],[s2])[0][0]
        ])
    X = np.array(X); y = np.array(labels)
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X,y)
    return clf, tfidf, idfbin

def build_feat(a_text, t_text, tfidf, idf):
    v1,v2 = tfidf.transform([a_text,t_text])
    i1,i2 = idf.transform([a_text,t_text])
    s1,s2 = style_vec(a_text), style_vec(t_text)
    return np.array([
        cosine_similarity(v1,v2)[0][0],
        cosine_similarity(i1,i2)[0][0],
        cosine_similarity([s1],[s2])[0][0]
    ]).reshape(1,-1)

# GUI
class App:
    def __init__(self, root):
        self.author_files = []; self.test_file = None
        root.title("Stylodel")
        root.geometry("380x240")
        tk.Button(root,text="Upload Samples of the Author ",width=36,command=self.load_author).pack(pady=10)
        tk.Button(root,text="Upload File to Analyze",width=36,command=self.load_test).pack(pady=10)
        tk.Button(root,text="Start verification",width=36,command=self.verify,fg="Green").pack(pady=12)
        self.status = tk.Label(root, text="Status: waiting", fg="blue"); self.status.pack()

    def load_author(self):
        files = filedialog.askopenfilenames(filetypes=[("Text","*.txt")])
        if files:
            self.author_files = list(files); self.status.config(text=f"{len(files)} Author A files selected")

    def load_test(self):
        f = filedialog.askopenfilename(filetypes=[("Text","*.txt")])
        if f: self.test_file = f; self.status.config(text=f"Test file: {os.path.basename(f)}")

    def verify(self):
        try:
            if not self.author_files or not self.test_file:
                messagebox.showerror("Error","Please upload both Author A samples and a test file."); return
            unknown = load_folder(UNKNOWN_FOLDER)
            if len(unknown) < 3:
                messagebox.showerror("Error", f"Put >=3 unknown .txt files into the folder '{UNKNOWN_FOLDER}'."); return
            self.status.config(text="Training meta-learner..."); self.root.update_idletasks()
            clf, tfidf, idf = train_meta(unknown)
            a_text = "\n".join(load_txt(p) for p in self.author_files)
            t_text = load_txt(self.test_file)
            feat = build_feat(a_text, t_text, tfidf, idf)
            prob = clf.predict_proba(feat)[0][1] * 100
            if prob>=70: res = f"LIKELY Author A ({prob:.2f}%)"
            elif prob>=40: res = f"DON'T KNOW ({prob:.2f}%)"
            else: res = f"LIKELY NOT Author A ({prob:.2f}%)"
            self.status.config(text="Done"); messagebox.showinfo("Result", res)
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    import tkinter as tk
    root = tk.Tk()
    app = App(root)
    app.root = root
    root.mainloop()
