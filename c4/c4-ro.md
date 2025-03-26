### 1: Introducere
- **Titlu:** „Învățare nesupervizată – tehnici de clustering”
- **Obiective:**
  - Elementele de bază ale clusteringului.
  - Familiarizare cu K-means, DBSCAN și Clusteringul Aglomerativ.
  - Evaluarea rezultatelor clusteringului.
  - Aplicații: detecția anomaliilor, segmentarea clienților, comprimarea imaginilor.

---

### 2: Învățarea nesupervizată
- **Definiție:** Învățarea tiparelor din date fără etichete.
- **Ideea de bază:** Detectarea unor structuri ascunse fără etichete explicite.
- **Tehnici comune:** Clustering, reducerea dimensionalității.
- **Focus principal:** Metode de clustering și aplicațiile lor.

---

### 3: Ce este clusteringul?
- **Definiție conceptuală:** Gruparea punctelor de date pe baza similarității.
- **Scop:** Maximizarea similarității în interiorul clusterului și minimizarea similarității între clustere diferite.
- **Scenarii de utilizare:** Explorarea datelor, analiză preliminară, segmentarea pieței.

---

### 4: Clustering K-Means - concepte
- **Pași de bază:**
  1. Alegerea numărului de clustere \( k \).
  2. Inițializarea centroizii clusterelor.
  3. Atribuirea punctele celui mai apropiat centroid.
  4. Recalculează centroizilor.
  5. Reperarea pașilor până la convergență.
- **Puncte forte:** Simplu, eficient pentru seturi mari de date.
- **Puncte slabe:** Necesită specificarea valorii \( k \), sensibil la outlieri, funcționează cel mai bine cu clustere sferice.

---

### 5: Exemplu K-Means
- **Procesul:**
  - Centroizi inițiali aleși aleator.
  - Atribuirea punctelor de date.
  - Actualizarea iterativă a centroidurilor.
- **[Un exemplu care ilustrează clusteringul K-means pe un set de date despre istoricul achizițiilor clienților]**

---

### 5.1: Metoda Elbow pentru K-Means
- **Motivație:** Cum se alege numărul optim de clustere \( k \).
- **Within-Cluster Sum of Squares (WCSS sau inerție):**
  - Se reprezintă WCSS în funcție de \( k \).
  - Caută „cotul” („elbow”) unde curba se nivelează vizibil.
- **Sfaturi practice:**
  - Cotul nu e întotdeauna clar.
  - Combină cu scorurile de tip siluetă sau cu cunoștințe de domeniu pentru o decizie mai solidă.
- **[Un exemplu care demonstrează graficul elbow cu diferite valori \( k \)]**

---

### 6: DBSCAN – concepte
- **Acronim:** Density-Based Spatial Clustering of Applications with Noise
- **Parametri cheie:**
  - \( \varepsilon \) (epsilon) – raza de vecinătate
  - \(\text{minPts}\) – numărul minim de puncte necesare pentru a forma o regiune densă
- **Beneficii:**
  - Poate găsi clustere de forme arbitrare.
  - Identifică outlieri ca zgomot.
- **Dezavantaje:**
  - Selectarea parametrilor poate fi dificilă.
  - Nu e potrivit pentru date cu dimensionalitate foarte mare.

---

### 7: Exemplu DBSCAN
- **Demonstrează formarea bazată pe densitate:**
  - Punctele din zone de densitate mare devin puncte „nucleu”.
  - Punctele din vecinătatea nucleului devin parte a clusterului.
  - Outlierii rămân neatribuiți.
- **[Un exemplu care arată cum DBSCAN separă clustere dense și etichetează punctele rare ca outlieri]**
- **[Articol](https://builtin.com/articles/dbscan#:~:text=What%20Is%20DBSCAN%3F-,Density%2Dbased%20spatial%20clustering%20of%20applications%20with%20noise%20(DBSCAN),data%20cleaning%20and%20outlier%20detection.)**

---

### 8: Clustering aglomerativ – Concepte
- **Abordare ierarhică:**
  - Fiecare punct începe ca propriul cluster.
  - Se unesc iterativ cele mai apropiate clustere.
  - Continuă până când toate punctele formează un singur cluster sau se atinge un criteriu de oprire.
- **Criterii de legătură (Linkage):** Single linkage, complete linkage, average linkage etc.
- **Dendrogramă:** Reprezentare vizuală a procesului de unire.

---

### 9: Exemplu de clustering aglomerativ
- **Interpretarea dendrogramei:**
  - Tăierea dendrogramei la diferite niveluri produce un număr diferit de clustere.
- **[Un exemplu care ilustrează cum se citește o dendrogramă pentru un set de imagini grupate după similaritatea vizuală]**

---

### 10: Evaluarea eezultatelor de clustering
- **Măsurători interne:**
  - Scorul siluetă, indexul Davies-Bouldin, indexul Calinski-Harabasz.
- **Măsurători externe (dacă sunt disponibile etichete):**
  - Rand index, Adjusted Rand index, Purity, F-measure.
- **Sfaturi practice:**
  - Folosește mai multe metrici pentru o imagine mai clară.
  - Inspecția vizuală este în continuare utilă pentru interpretare.

---

### 11: Aplicații – Detecția anomaliilor
- **Clustering pentru detectarea outlierilor:**
  - Outlierii sunt puncte care nu aparțin niciunui cluster (ex. DBSCAN).
  - Praguri de distanță în K-means pentru identificarea anomaliilor.
- **[Un exemplu care descrie cum traficul neobișnuit din rețea este marcat drept anomalie]**

---

### 12: Aplicații – segmentarea clienților
- **Obiectiv:** Gruparea clienților după similitudine, în vederea marketingului țintit.
- **Caracteristici tipice:** Istoric de achiziții, tipare de navigare, date demografice.
- **[Un exemplu care ilustrează cum segmentele sunt folosite pentru personalizarea emailurilor și ofertelor]**

---

### 13: Aplicații – Comprimarea imaginilor
- **Principiu:**
  - Clusterează pixelii într-un set mai mic de culori (ex. K-means).
  - Înlocuiește fiecare pixel cu culoarea reprezentativă a clusterului.
- **[Un exemplu care arată cum o fotografie este comprimată prin reducerea paletei de culori fără pierderi vizuale majore]**

---

### 14: Q&A și Concluzii
- **Puncte cheie:**
  - Cele trei metode de clustering (K-means, DBSCAN, Aglomerativ).
  - Evaluarea clusterelor (metrici interne vs. externe).
  - Utilizări practice în detecția anomaliilor, segmentarea clienților, comprimarea imaginilor.
- **Pași următori:**
  - Experimentează cu diferite algoritmi și seturi de date.
  - Explorează tehnici avansate de clustering (spectral clustering etc.).
