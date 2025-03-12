###  1: Prezentare generală
- **Titlu**: Învățare supervizată – clasificare
- **Structura cursului**:
  - Definiția problemei de clasificare
  - Naïve Bayes
  - Regressie logistică
  - Arbori de clasificare
  - Overfitting & regularizare (Lasso, Ridge)
  - Metrici de evaluare: accuracy, precision-recall, ROC-AUC
  - Aplicații în lumea reală

---

###  2: Introducere în problemele de clasificare
- **Definiție**: Prezicerea unei etichete sau categorii discrete
- **Exemple**:
  - Spam sau non-spam
  - Pacient bolnav sau sănătos
- **De ce clasificare?**
  - Folosită frecvent în diagnostic medical, filtre de e-mail, detectarea fraudei

---

###  3: Concepte cheie pentru clasificare
- **Caracteristici & etichete**:
  - Caracteristici = variabile de intrare (predictori)
  - Etichete = clasele țintă
- **Antrenare vs. Testare**:
  - Antrenare pe date etichetate
  - Testare pentru validarea performanței
- **[Exemplu: Prezintă un set de date mic cu caracteristici precum vârsta, venitul și o etichetă binară care indică „purchased” sau „not purchased.”]**

---

###  4: Clasificatorul Naïve Bayes
- **Ideea de bază**:
  - Utilizează teorema lui Bayes cu presupunerea independenței caracteristicilor
- **Avantaje**:
  - Simplu, rapid, funcționează bine cu date zgomotoase
- **Dezavantaje**:
  - Presupunerea independenței poate fi nerealistă în anumite domenii
- **[Exemplu: Clasificarea recenziilor ca pozitive sau negative pe baza frecvenței cuvintelor]**
- **[Articol](https://towardsdatascience.com/multinomial-naive-bayes-for-documents-classification-and-natural-language-processing-nlp-e08cc848ce6/)**
---

###  5: Regressie logistică
- **Concept**:
  - Modelează probabilitatea ca o anumită clasă să fie rezultatul
  - Utilizează funcția logistică (grafic sigmoid)
- **Interpretabilitate**:
  - Coeficienții pot fi analizați pentru a vedea impactul caracteristicilor
- **Când se utilizează**:
  - Clasificare binară cu o graniță de decizie liniară
- **[Exemplu: Prezicerea probabilității ca un client să facă o achiziție pe baza datelor demografice]**
- **[Articol](https://datatab.net/tutorial/logistic-regression)**


---

###  6: Arbori de clasificare
- **Structură**:
  - Serie de reguli decizionale bazate pe valorile caracteristicilor
- **Avantaje**:
  - Ușor de vizualizat și interpretat
- **Dezavantaje**:
  - Predispus la suprapotrivire dacă nu este tăiat (pruned)
- **[Exemplu: Un arbore de decizie pentru aprobarea unui împrumut pe baza caracteristicilor precum venit, scor de credit și raportul datorii/venit]**
- **[Articol](https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html)**
---

###  7: Overfitting & regularizare
- **Suprapotrivire**:
  - Modelul se potrivește zgomotului sau detaliilor specifice din datele de antrenare
  - Simptome: acuratețe mare pe antrenare, generalizare slabă
- **Regularizare**:
  - Penalizează coeficienții mari pentru a reduce overfitting
  - Metode comune: Lasso (L1), Ridge (L2)
- **[Articol](https://www.analyticsvidhya.com/blog/2021/09/lasso-and-ridge-regularization-a-rescuer-from-overfitting/)

---

###  8: Lasso (L1) vs. Ridge (L2)
- **Lasso (L1)**:
  - Poate aduce coeficienții la zero → selecția caracteristicilor
- **Ridge (L2)**:
  - Coeficienții sunt micșorați spre zero, dar rareori devin zero
- **Când se utilizează**:
  - Lasso: preferi modele mai simple, cu mai puține caracteristici
  - Ridge: vrei să păstrezi toate caracteristicile, dar să reduci variația mare
- **[Exemplu: Aplicarea regresiei logistice cu regularizare L1 sau L2 pe un set de date cu multe caracteristici și observarea caracteristicilor semnificative]**
- **[Articol]((https://www.analyticsvidhya.com/blog/2021/07/metrics-to-evaluate-your-classification-model-to-take-the-right-decisions/))**

---

###  9: Metrici de evaluare
- **Accuracy**:
  - Proporția etichetelor prezise corect
  - Utilă când clasele sunt echilibrate
- **Precision & recall**:
  - Precision = dintre predicțiile pozitive, câte sunt corecte
  - Recall = dintre cazurile pozitive reale, câte au fost identificate
- **ROC & AUC**:
  - Graficul ratei adevărat pozitive vs. ratei fals pozitive
  - ROC = Receiver-operating characteristic curve
  - AUC = Aria sub curba ROC (măsură a performanței generale)
- **[Exemplu: Prezintă o matrice de confuzie pentru o problemă de clasificare binară și calculează precizia, recall-ul și acuratețea]**
- **[Articol](https://towardsdatascience.com/understanding-the-roc-curve-and-auc-dd4f9a192ecb/)**

---

###  10: Alegerea metricii corecte
- **Considerații cheie**:
  - Dezechilibru de clase: acuratețea poate fi înșelătoare
  - Costul fals pozitive vs. fals negative
  - Raportarea mai multor metrici pentru claritate
- **[Exemplu: Set de date dezechilibrat în domeniul sănătății, unde fals negativele pot fi foarte costisitoare]**

---

###  11: Aplicații reale
- **Clasificare de text**:
  - Detectarea spamului, analiza sentimentelor
- **Diagnostic medical**:
  - Clasificarea bolilor, evaluarea riscurilor
- **Domeniul financiar**:
  - Scor de credit, detectarea fraudei
- **[Exemplu: Cum folosește o bancă regresia logistică pentru a identifica tranzacțiile frauduloase]**

---

###  12: Concluzii
- **Rezumat**:
  - Clasificarea este centrală pentru multe sarcini predictive
  - Există mai multe algoritmi, fiecare cu puncte forte și slabe
  - Măsurile de evaluare ghidează selecția modelului
- **Următorii pași**:
  - Experimentează cu diferite tehnici de clasificare
  - Concentrează-te pe regularizare și selecția corectă a metricilor
  - Explorează modele avansate de clasificare (ex. Random Forest, SVM)
- **Întrebări / Discuții**
