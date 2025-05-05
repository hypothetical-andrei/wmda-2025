### **Slide 1: Introducere (5 minute)**
- **Titlu:** “Rețele Neurale – Fundamente & Antrenare”
- **Subtitlu:** Cursul 5 din Seria de Învățare Automată
- **Obiective:**
  - Înțelegerea structurii și antrenării rețelelor neurale.
  - Construirea unei rețele folosind TensorFlow și PyTorch.
  - Învățarea conceptelor de bază: perceptroni, funcții de activare, backpropagation.
  - Explorarea optimizării, regularizării și sfaturilor practice.

---

### **Slide 2: Introducere în rețelele neuronale (5 minute)**
- **Definiție:** Rețelele neurale sunt o serie de algoritmi care imită creierul uman pentru a recunoaște tipare.
- **Structură:** Formate din straturi de „neuroni” (noduri) conectați.
- **De ce este important:** Fundament pentru învățarea profundă și aplicațiile AI.
- **Cazuri de utilizare comune:** Clasificare de imagini, NLP, jocuri.

---

### **Slide 3: TensorFlow vs. PyTorch (5 minute)**
- **TensorFlow:**
  - Grafuri de calcul statice.
  - Excelent pentru implementare și performanță.
- **PyTorch:**
  - Grafuri dinamice.
  - Mai intuitiv și mai „pythonic”; preferat în cercetare.
- **[Un exemplu care arată cod echivalent pentru definirea unei rețele neurale în TensorFlow](https://github.com/hypothetical-andrei/wmda-2025/blob/main/c5/example1a.py)** și **[PyTorch](https://github.com/hypothetical-andrei/wmda-2025/blob/main/c5/example1b.py)**

---

### **Slide 4: Perceptronul – cea mai simplă rețea (10 minute)**
- **Concept:**
  - Rețea neurală cu un singur strat.
  - Clasificator binar care mapează intrările în ieșiri prin greutăți.
- **Cum funcționează:**
  - Sumă ponderată → activare → predicție.
- **[Un exemplu: Utilizarea unui perceptron pentru a clasifica puncte deasupra sau dedesubtul unei linii](https://github.com/hypothetical-andrei/wmda-2025/blob/main/c5/example2.py)**

---

### **Slide 5: Construirea rețelelor neuronale de la zero (5 minute)**
- **Perceptron Multistrat (MLP):**
  - Strat de intrare → straturi ascunse → strat de ieșire.
  - Adâncimea și dimensiunile straturilor afectează capacitatea modelului.
- **[Un exemplu: Implementarea manuală a unui MLP pentru clasificarea cifrelor scrise de mână](https://github.com/hypothetical-andrei/wmda-2025/blob/main/c5/example3.py)**

---

### **Slide 6: Backpropagation & gradient descent (10 minute)**
- **Backpropagation:**
  - Calculează cum ar trebui ajustate greutățile pe baza erorilor.
- **Gradient Descent:**
  - Metodă de optimizare pentru minimizarea pierderii prin actualizarea greutăților.
- **Variante:** Batch, stochastic, mini-batch.
- **[Un exemplu care arată cum scade pierderea în timpul antrenării cu gradient descent](https://github.com/hypothetical-andrei/wmda-2025/blob/main/c5/example4.py)**

---

### **Slide 7: Antrenarea unei rețelr neuronale (5 minute)**
- **Pași:**
  1. Trecere înainte (forward pass) pentru calculul ieșirii.
  2. Calcularea pierderii.
  3. Backpropagation pentru obținerea gradientelor.
  4. Optimizatorul actualizează greutățile.
- **[Un exemplu: Buclă de antrenare în PyTorch pe un set de date simplu](https://github.com/hypothetical-andrei/wmda-2025/blob/main/c5/example5.py)**

---

### **Slide 8: Funcții de activare (10 minute)**
- **De ce sunt necesare:** Introduc neliniaritate pentru mapări complexe de funcții.
- **Tipuri comune:**
  - **ReLU:** Eficientă, implicită pentru majoritatea rețelelor.
  - **Sigmoid:** Aplatizează valorile, bună pentru probabilități.
  - **Tanh:** Centrată pe zero, similară cu sigmoidul.
  - **Softmax:** Probabilități pentru ieșiri multi-clasă.
- **[Un exemplu care vizualizează ieșirea fiecărei funcții de activare pe date de probă](https://github.com/hypothetical-andrei/wmda-2025/blob/main/c5/example6.py)**

---

### **Slide 9: Rate de învățare, optimizatori & funcții de calcul al erorii (10 minute)**
- **Rata de învățare:** Controlează dimensiunea pașilor de actualizare.
- **Optimizatori:**
  - SGD, Adam, RMSprop.
  - Influențează viteza și stabilitatea convergenței.
- **Funcții de calcul al erorii:**
  - MSE, Entropie Încrucișată (binare & categorice).
- **[Un exemplu care compară diverși optimizatori pe o sarcină de clasificare](https://github.com/hypothetical-andrei/wmda-2025/blob/main/c5/example7.py)**

---

### **Slide 10: Supraantrenare & regularizare (10 minute)**
- **Supraantrenare:** Modelul merge bine pe datele de antrenare dar prost pe datele nevăzute.
- **Tehnici:**
  - **Dropout:** Dezactivează aleatoriu neuroni în timpul antrenării.
  - **Batch normalization:** Normalizează intrările fiecărui strat.
- **Alte instrumente:** Early stopping, penalizare a greutăților (weight decay).
- **[Un exemplu care arată pierderea pe antrenare vs validare cu și fără dropout](https://github.com/hypothetical-andrei/wmda-2025/blob/main/c5/example8.py)**

---

### **Slide 11: Rezumat & întrebări (5 minute)**
- **Recapitulare concepte-cheie:**
  - Structura rețelei neurale și bucla de antrenare.
  - Funcții de activare și optimizatori.
  - Regularizare pentru a preveni supraantrenarea.
- **Pașii următori:**
  - Exersați implementarea rețelelor mici.
  - Explorați rețelele neuronale convoluționale (CNN) în sesiunea următoare.
