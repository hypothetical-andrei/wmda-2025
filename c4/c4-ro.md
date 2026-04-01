# Învățare nesupervizată – tehnici de clustering

## Introducere

**Obiective:**

- Elementele de bază ale clusteringului
- Familiarizare cu K-means, DBSCAN și Clusteringul Aglomerativ
- Evaluarea rezultatelor clusteringului
- Aplicații: detecția anomaliilor, segmentarea clienților, comprimarea imaginilor

## Învățarea nesupervizată

- **Definiție:** Învățarea tiparelor din date fără etichete
- **Ideea de bază:** Detectarea unor structuri ascunse fără etichete explicite
- **Tehnici comune:** Clustering, reducerea dimensionalității
- **Focus principal:** Metode de clustering și aplicațiile lor

## Ce este clusteringul?

- **Definiție conceptuală:** Gruparea punctelor de date pe baza similarității
- **Scop:** Maximizarea similarității în interiorul clusterului și minimizarea similarității între clustere diferite
- **Scenarii de utilizare:** Explorarea datelor, analiză preliminară, segmentarea pieței

## Clustering K-Means – Concepte

**Pași de bază:**

1. Alegerea numărului de clustere $k$
2. Inițializarea centroizilor clusterelor
3. Atribuirea punctelor celui mai apropiat centroid
4. Recalcularea centroizilor
5. Repetarea pașilor până la convergență

- **Puncte forte:** Simplu, eficient pentru seturi mari de date
- **Puncte slabe:** Necesită specificarea valorii $k$, sensibil la outlieri, funcționează cel mai bine cu clustere sferice

## Exemplu K-Means

**Procesul:**

- Centroizi inițiali aleși aleator
- Atribuirea punctelor de date
- Actualizarea iterativă a centroidurilor

*[Exemplu 1 – Ilustrează clusteringul K-means pe un set de date despre istoricul achizițiilor clienților — `example1.py`]*

## Metoda Elbow pentru K-Means

- **Motivație:** Cum se alege numărul optim de clustere $k$
- **Within-Cluster Sum of Squares (WCSS sau inerție):**
  - Se reprezintă WCSS în funcție de $k$
  - Caută „cotul" („elbow") unde curba se nivelează vizibil
- **Sfaturi practice:**
  - Cotul nu e întotdeauna clar
  - Combină cu scorurile de tip siluetă sau cu cunoștințe de domeniu pentru o decizie mai solidă

*[Exemplu 1a – Demonstrează graficul elbow cu diferite valori $k$ — `example1a.py`]*

## DBSCAN – Concepte

**Acronim:** Density-Based Spatial Clustering of Applications with Noise

**Parametri cheie:**

- $\varepsilon$ (epsilon) – raza de vecinătate
- $\text{minPts}$ – numărul minim de puncte necesare pentru a forma o regiune densă

- **Beneficii:** Poate găsi clustere de forme arbitrare; identifică outlieri ca zgomot
- **Dezavantaje:** Selectarea parametrilor poate fi dificilă; nu e potrivit pentru date cu dimensionalitate foarte mare

## Exemplu DBSCAN

**Demonstrează formarea bazată pe densitate:**

- Punctele din zone de densitate mare devin puncte „nucleu"
- Punctele din vecinătatea nucleului devin parte a clusterului
- Outlierii rămân neatribuiți

*[Exemplu 2 – Arată cum DBSCAN separă clustere dense și etichetează punctele rare ca outlieri — `example2.py`]*

## Clustering Aglomerativ – Concepte

**Abordare ierarhică:**

- Fiecare punct începe ca propriul cluster
- Se unesc iterativ cele mai apropiate clustere
- Continuă până când toate punctele formează un singur cluster sau se atinge un criteriu de oprire

- **Criterii de legătură (Linkage):** Single linkage, complete linkage, average linkage etc.
- **Dendrogramă:** Reprezentare vizuală a procesului de unire

## Exemplu de Clustering Aglomerativ

**Interpretarea dendrogramei:**

- Tăierea dendrogramei la diferite niveluri produce un număr diferit de clustere

*[Exemplu 3 – Ilustrează cum se citește o dendrogramă pentru un set de imagini grupate după similaritatea vizuală — `example3.py`]*

## Evaluarea rezultatelor de clustering

- **Măsurători interne** (nu necesită etichete): Scorul siluetă, indexul Davies-Bouldin, indexul Calinski-Harabasz
- **Măsurători externe** (necesită etichete): Rand index, Adjusted Rand index, Purity, F-measure
- **Sfaturi practice:**
  - Folosește mai multe metrici pentru o imagine mai clară
  - Inspecția vizuală este în continuare utilă pentru interpretare

## Scorul Siluetă

Pentru un punct $i$:

$s(i) = \frac{b(i) - a(i)}{\max(a(i),\ b(i))}$

- $a(i)$ = distanța medie față de celelalte puncte din **același cluster** (coeziune)
- $b(i)$ = distanța medie minimă față de punctele din **cel mai apropiat cluster vecin** (separare)

Scorul global: $S = \frac{1}{n} \sum_{i=1}^{n} s(i)$, cu valori în $[-1, 1]$ — mai aproape de $1$ este mai bun.

*[Exemplu 3a — `example3a.py`]*

## Indexul Davies-Bouldin

$DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{s_i + s_j}{d(c_i, c_j)} \right)$

- $k$ = numărul de clustere
- $s_i$ = dispersia medie a clusterului $i$ (distanța medie a punctelor față de centroidul lor)
- $d(c_i, c_j)$ = distanța dintre centroizii clusterelor $i$ și $j$

Valori **mai mici** indică clustere mai compacte și mai bine separate.

*[Exemplu 3a — `example3a.py`]*

## Indexul Calinski-Harabasz

$CH = \frac{SS_B / (k-1)}{SS_W / (n-k)}$

- $SS_B = \sum_{i=1}^{k} n_i \|c_i - c\|^2$ — dispersia **inter**-cluster
- $SS_W = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - c_i\|^2$ — dispersia **intra**-cluster
- $n_i$ = numărul de puncte din clusterul $i$, $c$ = centroidul global

Valori **mai mari** indică un clustering mai bun.

*[Exemplu 3a — `example3a.py`]*

## Rand Index și Adjusted Rand Index

Pentru orice pereche de puncte $(i, j)$, se numără asocierile cu etichetele reale:

$RI = \frac{TP + TN}{TP + TN + FP + FN}$

Adjusted Rand Index corectează pentru asocierile întâmplătoare:

$ARI = \frac{RI - \mathbb{E}[RI]}{\max(RI) - \mathbb{E}[RI]}$

- $RI \in [0, 1]$, $ARI \in [-1, 1]$ — valori mai mari sunt mai bune
- $ARI = 1$ înseamnă acord perfect cu etichetele reale

*[Exemplu 3a — `example3a.py`]*

## Purity și F-measure

**Purity** — pentru fiecare cluster, fracția punctelor din clasa dominantă:

$Purity = \frac{1}{n} \sum_{i=1}^{k} \max_{j} |C_i \cap L_j|$

**F-measure** — media armonică a preciziei și a recall-ului la nivel de cluster:

$F = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$

- Ambele în $[0, 1]$ — valori mai mari sunt mai bune
- Purity nu penalizează numărul mare de clustere; F-measure este mai echilibrat

*[Exemplu 3a — `example3a.py`]*

## Comparație metrici interne

| Metrică | Optim | Interval |
|---|---|---|
| Silhouette | maxim | $[-1, 1]$ |
| Davies-Bouldin | minim | $[0, \infty)$ |
| Calinski-Harabasz | maxim | $[0, \infty)$ |

## Optimizarea parametrilor DBSCAN

- Metricile interne pot ghida căutarea parametrilor $\varepsilon$ și $\text{minPts}$
- GridSearch cu scor siluetă ca funcție obiectiv

*[Exemplu 3b – Optimizarea parametrilor DBSCAN cu scor siluetă prin GridSearch — `example3b.py`]*

## Aplicații – Detecția anomaliilor

- **Clustering pentru detectarea outlierilor:**
  - Outlierii sunt puncte care nu aparțin niciunui cluster (ex. DBSCAN)
  - Praguri de distanță în K-means pentru identificarea anomaliilor

*[Exemplu 4 – Descrie cum traficul neobișnuit din rețea este marcat drept anomalie — `example4.py`]*

## Aplicații – Segmentarea clienților

- **Obiectiv:** Gruparea clienților după similitudine, în vederea marketingului țintit
- **Caracteristici tipice:** Istoric de achiziții, tipare de navigare, date demografice

*[Exemplu 5 – Ilustrează cum segmentele sunt folosite pentru personalizarea emailurilor și ofertelor — `example5.py`]*

## Aplicații – Comprimarea imaginilor

**Principiu:**

- Clusterează pixelii într-un set mai mic de culori (ex. K-means)
- Înlocuiește fiecare pixel cu culoarea reprezentativă a clusterului

*[Exemplu 6 – Arată cum o fotografie este comprimată prin reducerea paletei de culori fără pierderi vizuale majore — `example6.py`]*

## Q&A și Concluzii

**Puncte cheie:**

- Cele trei metode de clustering (K-means, DBSCAN, Aglomerativ)
- Evaluarea clusterelor (metrici interne vs. externe)
- Utilizări practice în detecția anomaliilor, segmentarea clienților, comprimarea imaginilor

**Pași următori:**

- Experimentează cu diferiți algoritmi și seturi de date
- Explorează tehnici avansate de clustering (spectral clustering etc.)