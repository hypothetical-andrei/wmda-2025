### **Tutorial: Procesarea Practică a Datelor & Formularea Problemei în AI**
#### **Exerciții (10 minute fiecare)**

---

### **Exercițiul 1: Extracția și curățarea Datelor dintr-un API**
**Obiectiv:** Obțineți și curățați date reale dintr-un API public.

#### **Pași:**
1. Obțineți date meteo folosind API-ul OpenWeatherMap (fără cheie API, utilizați: `https://wttr.in/?format=%C+%t`).
2. Parcurgeți răspunsul API și extrageți informațiile relevante (ex: temperatură, condiții meteo).
3. Curățați datele eliminând caracterele inutile și formatați-le într-un **DataFrame Pandas** structurat.
4. Salvați setul de date curățat ca fișier CSV.

**Rezultat așteptat:** Un **DataFrame** structurat cu coloanele `Oraș`, `Temperatură`, `Condiție Meteo`.

---

### **Exercițiul 2: Web scraping a unei pagini cu produse**
**Obiectiv:** Extrageți și structurați date de pe un site web public.

#### **Pași:**
1. Utilizați **BeautifulSoup** pentru a extrage numele produselor și prețurile de pe un site de testare e-commerce: [Web Scraper Test Site](https://webscraper.io/test-sites/e-commerce/allinone/computers/laptops).
2. Extrageți **numele produselor** și **prețurile** de pe pagină.
3. Stocați datele extrase într-un **DataFrame Pandas**.
4. Eliminați eventualele produse duplicate.

**Rezultat așteptat:** Un **DataFrame** cu coloanele `Nume Produs` și `Preț`, conținând cel puțin 10 produse.

---

### **Exercițiul 3: Implementarea unui sistem simplu de recomandare**
**Obiectiv:** Construiți un sistem de recomandare bazat pe **filtrare colaborativă** folosind dataset-ul **MovieLens**.

#### **Pași:**
1. Încărcați **setul de date MovieLens 100K** (fișierul `u.data`) folosind Pandas.
2. Preprocesați datele filtrând utilizatorii care au evaluat mai puțin de **10 filme**.
3. Calculați **ratingul mediu per film** și sortați filmele după popularitate.
4. Recomandați **top 5 cele mai populare filme** pentru utilizatorii noi.

**Rezultat așteptat:** O listă clasificată cu **top 5 filme** pe baza ratingurilor utilizatorilor.

---

### **Exercițiul 4: Feature engineering pentru clasificare**
**Obiectiv:** Creați noi caracteristici și normalizați-le pentru algoritmi de învățare automată.

#### **Pași:**
1. Încărcați **setul de date Titanic** din Seaborn (`sns.load_dataset('titanic')`).
2. Creați o **nouă caracteristică**: `family_size = sibsp + parch + 1` (Numărul total de membri ai familiei la bord).
3. Codificați variabilele categorice (`sex`, `embarked`) folosind **one-hot encoding**.
4. Normalizați caracteristicile numerice (`age`, `fare`, `family_size`) folosind **MinMaxScaler**.

**Rezultat așteptat:** Un **DataFrame** curățat și transformat, pregătit pentru clasificare.

---

### **Exercițiul 5: Aplicarea unui model de clasificare**
**Obiectiv:** Antrenați un model simplu de clasificare folosind setul de date Titanic.

#### **Pași:**
1. Utilizați setul de date Titanic preprocesat din **Exercițiul 4**.
2. Împărțiți datele în **seturi de antrenare și testare** (`train_test_split`).
3. Antrenați un model de **Regresie Logistică** pentru a prezice supraviețuirea.
4. Evaluați modelul folosind **acuratețe, precizie și recall**.

**Rezultat așteptat:** Un raport de clasificare care arată performanța modelului.

---

### **Exercițiu bonus: Ajustarea hiperparametrilor pentru clasificare**
**Obiectiv:** Îmbunătățiți performanța modelului folosind **ajustarea hiperparametrilor**.

#### **Pași:**
1. Utilizați **GridSearchCV** pentru a găsi cei mai buni parametri pentru modelul de **Regresie Logistică**.
2. Reglați `C` (forța regularizării) și `penalty` (L1 vs. L2).
3. Comparați acuratețea înainte și după reglaj.

**Rezultat așteptat:** Un rezumat al celor mai buni hiperparametri și îmbunătățirea performanței modelului.
