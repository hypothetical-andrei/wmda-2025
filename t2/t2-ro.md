### Exercițiul 1 (10 minute): Clasificare Naïve Bayes

1. **Obiectiv**
   - Încarcă un set de date mic pentru clasificare din scikit-learn (de exemplu, setul de date Iris sau Wine).
   - Implementează un clasificator Naïve Bayes.

2. **Instrucțiuni**
   1. Importă bibliotecile necesare (`pandas`, `numpy`, `sklearn`).
   2. Încarcă setul de date folosind `sklearn.datasets.load_wine()` (sau alt set de date).
   3. Împarte setul de date în seturi de antrenare și testare (utilizează o împărțire de 80/20).
   4. Antrenează un clasificator Naïve Bayes (`sklearn.naive_bayes.GaussianNB`).
   5. Prezice pe setul de testare și afișează scorul de acuratețe.

3. **Puncte cheie de verificat**
   - Datele sunt împărțite corect?
   - Modelul se antrenează fără erori?
   - Observi o acuratețe rezonabilă?

4. **Extensie posibilă** (dacă rămâne timp)
   - Afișează și interpretează matricea de confuzie pentru o înțelegere mai profundă a performanței.

---

### Exercițiul 2 (10 minute): Regresie logistică și compararea modelelor

1. **Obiectiv**
   - Implementează regresia logistică pe același set de date din Exercițiul 1.
   - Compară performanța cu Naïve Bayes.

2. **Instrucțiuni**
   1. Reutilizează setul de date și împărțirea train/test din Exercițiul 1.
   2. Antrenează un model de regresie logistică (`sklearn.linear_model.LogisticRegression`).
   3. Calculează acuratețea, precizia și recall-ul.
   4. Compară aceste valori cu rezultatele obținute de modelul Naïve Bayes.

3. **Puncte cheie de verificat**
   - Regresia logistică are performanțe mai bune decât Naïve Bayes sau invers? (Gândește-te de ce.)
   - Apar avertismente privind convergența în regresia logistică? (Poate fi necesară ajustarea `max_iter`).

---

### Exercițiul 3 (10 minute): Arbori de decizie

1. **Obiectiv**
   - Implementează un clasificator cu arbori de decizie și vizualizează/interpretază structura acestuia.

2. **Instrucțiuni**
   1. Folosește aceeași împărțire a datelor (sau creează una nouă, dacă preferi).
   2. Antrenează un arbore de decizie (`sklearn.tree.DecisionTreeClassifier`).
   3. Verifică acuratețea pe setul de testare.
   4. Utilizează `sklearn.tree.plot_tree` sau `export_graphviz` (dacă este disponibil) pentru a vizualiza structura arborelui (sau cel puțin analizează `feature_importances_` pentru a vedea care caracteristici sunt cele mai relevante).

3. **Puncte cheie de verificat**
   - Evită supraînvățarea (mai ales dacă adâncimea maximă nu este restricționată).
   - Observă câți noduri/frunze sunt create.

4. **Extensie posibilă** (dacă rămâne timp)
   - Restricționează adâncimea maximă a arborelui și vezi dacă performanța se modifică.

---

### Exercițiul 4 (10 minute): Clasificarea spam-ului cu Scikit-learn

1. **Obiectiv**
   - Aplică abilitățile de clasificare pe un set de date real de tip spam vs. non-spam.
   - Utilizează setul de date Spambase (disponibil aici: [Spambase dataset](https://archive.ics.uci.edu/dataset/94/spambase)).

2. **Instrucțiuni**
   1. Încarcă un set de date despre spam (de exemplu, colecția publică „SMS Spam Collection” sau un set de date similar).
   2. Convertește textul în caracteristici numerice (utilizând `CountVectorizer` sau `TfidfVectorizer`).
   3. Împarte datele în seturi de antrenare și testare.
   4. Antrenează unul dintre cei trei clasificatori (Naïve Bayes, Regresie Logistică sau Arbore de Decizie) pentru a prezice spam vs. non-spam.
   5. Afișează acuratețea și matricea de confuzie.

3. **Puncte cheie de verificat**
   - Vectorizarea textului este corectă? (Se elimină stopword-urile, dacă este necesar?)
   - Compară performanța între diferiți clasificatori (dacă ai timp).

4. **Extensie posibilă** (dacă rămâne timp)
   - Evaluează precizia și recall-ul special pentru clasa spam (clasa pozitivă).

---

### Exercițiul 5 (10 minute): Analiza Sentimentului pe Recenzii de Filme

1. **Obiectiv**
   - Realizează clasificarea sentimentului pe un set mic de recenzii de filme (pozitiv vs. negativ).

2. **Instrucțiuni**
   1. Încarcă un subset de recenzii de filme etichetate „pozitiv” sau „negativ” (de exemplu, `nltk.corpus.movie_reviews`, dacă este disponibil).
   2. Transformă textul în caracteristici numerice (`TfidfVectorizer` este adesea mai eficient pentru analiza sentimentului).
   3. Antrenează un model de regresie logistică pentru clasificare (sau alt clasificator, dacă preferi).
   4. Evaluează acuratețea și scorul F1 al modelului.

3. **Puncte cheie de verificat**
   - Există anumite cuvinte care influențează puternic clasificarea? (Opțional: verifică `coef_` în regresia logistică.)
   - Distribuția claselor este echilibrată?

4. **Extensie posibilă** (dacă rămâne timp)
   - Compară rezultatele cu un Arbore de Decizie sau Naïve Bayes pentru a vedea care model funcționează mai bine pe text.

---

### Exercițiul 6 (10 minute): Validare încrucișată și reglarea hiperparametrilor

1. **Obiectiv**
   - Utilizează validarea încrucișată și o căutare simplă de hiperparametri pentru a îmbunătăți performanța modelului.

2. **Instrucțiuni**
   1. Alege una dintre sarcinile de clasificare anterioare (analiza spam sau sentimentul sunt recomandate).
   2. Folosește `GridSearchCV` sau `RandomizedSearchCV` pentru a regla hiperparametrii (de exemplu, `alpha` pentru Naïve Bayes, `C` pentru Regresia Logistică, `max_depth` pentru Arbori de Decizie).
   3. Rulează o căutare cu validare încrucișată pentru a găsi cei mai buni parametri.
   4. Afișează cei mai buni parametri și cel mai bun scor obținut în validarea încrucișată.

3. **Puncte cheie de verificat**
   - Configurarea corectă a grilei de parametri.
   - Timpul de rulare (recomandat să fie o căutare rapidă pentru exercițiu de 10 minute).