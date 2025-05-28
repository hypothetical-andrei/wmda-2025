### 1: Prezentare generală
- **Titlu**: NLP cu Rețele Neuronale – Transformere & Modelarea Secvențelor
- **Agendă**:
  1. Reprezentări vectoriale ale cuvintelor (Word2Vec, GloVe)
  2. Rețele neuronale recurente (RNN, LSTM, GRU)
  3. Mecanisme de atenție & Transformere
  4. BERT, GPT și arhitecturi moderne NLP
  5. Aplicații în analiza sentimentelor, rezumarea textului și chatbots
- **Obiectiv**: Înțelegerea conceptelor-cheie și a utilizărilor practice ale arhitecturilor moderne NLP.

---

### 2: Introducere în reprezentările vectoriale ale cuvintelor
- **Concept**: Transformarea cuvintelor în vectori care surprind semnificația semantică
- **Beneficii**:
  * Reprezentare mai bună a cuvintelor
  * Dimensionalitate redusă a datelor text
- **Exemple**:
  * Similarități între cuvinte (ex.: „rege” este mai apropiat de „regină” decât de „măr”)
  * [Exemplu: Afișează un set mic de vectori pentru cuvintele „rege”, „regină”, „bărbat”, „femeie”, demonstrând distanțele și analogiile](https://github.com/hypothetical-andrei/wmda-2025/blob/main/c6/example1.py)

---

### 3: Word2Vec & GloVe
- **Word2Vec**:
  * Arhitecturi Skip-gram și CBOW (Continuous Bag of Words)
  * Învață reprezentări vectoriale prezicând cuvintele vecine
- **GloVe**:
  * Vectori Globali pentru reprezentarea cuvintelor
  * Folosește statistici de co-apariție din întregul corpus
- **Comparație**:
  * Word2Vec = model predictiv, GloVe = model bazat pe numărare
  * Ambele generează reprezentări care surprind relații semantice
  * [Exemplu: Afișează un fragment de cod pentru antrenarea unui model simplu Word2Vec pe un corpus mic de text](https://github.com/hypothetical-andrei/wmda-2025/blob/main/c6/example2.py)

---

### 4: Rețele Neuronale Recurente
- **Bazele RNN**:
  * Proiectate pentru date secvențiale (text, serii de timp, înregistrari audio ale limbajului)
  * Starea ascunsă captează informații din trecut
  * Provocări: gradienti care dispar/explodează
- **LSTM (Long Short-Term Memory)**:
  * Introduce starea celulei + porți pentru a atenua problemele cu gradientul care dispare
  * Mai eficient la captarea dependențelor de lungă durată
- **GRU (Gated Recurrent Unit)**:
  * Versiune simplificată a LSTM cu mai puțini parametri
  * [Exemplu: Demonstrează antrenarea unui LSTM pentru analiza sentimentelor pe un set mic de date](https://github.com/hypothetical-andrei/wmda-2025/blob/main/c6/example3.py)

---

### 5: Mecanisme de atenție
- **Motivație**:
  * RNN-urile au dificultăți cu dependențele de lungă durată
  * Atenția permite modelului să se concentreze pe părțile relevante ale secvenței
- **Ideea Principală**:
  * Calculează o sumă ponderată a stărilor ascunse
  * Ponderi mai mari pentru cuvintele mai importante
- **Beneficii**:
  * Îmbunătățește performanța în traducere, rezumare etc.
  * [Exemplu: Un model simplu „toy” secvență-la-secvență cu atenție, vizualizând ponderile atenției](https://github.com/hypothetical-andrei/wmda-2025/blob/main/c6/example4.py)

---

### 6: Arhitectura Transformator
- **Transformatori**:
  * Bazați complet pe atenție (fără convoluții sau conexiuni recurente)
  * Structură Encoder-Decoder
  * Mecanism de auto-atenție
- **Avantaje**:
  * Antrenare paralelizabilă
  * Gestionează eficient secvențe lungi
- **Componente Cheie**:
  * Atenție multi-head, poziționare vectorială, straturi feed-forward
  * [Exemplu: Afișează o diagramă care evidențiază straturile de auto-atenție ale encoderului și modul în care acestea se concentrează pe diferite cuvinte dintr-o propoziție](https://github.com/hypothetical-andrei/wmda-2025/blob/main/c6/example5.py)

---

### 7: BERT, GPT & NLP Modern
- **BERT (Bidirectional Encoder Representation from Transformers)**:
  * Utilizează context bidirecțional
  * Antrenat pe modelare de limbaj cu cuvinte mascate
- **GPT (Generalized Pretrained Transformer)**:
  * Model autoregresiv (de la stânga la dreapta)
  * Foarte eficient la generarea de text
- **Alte arhitecturi**:
  * T5, XLNet, ALBERT etc.
  * Fine-tuning pentru sarcini specifice
  * [Exemplu: Compară rezultatele BERT și GPT în completarea propozițiilor cu cuvinte lipsă sau continuarea unui început de text](https://github.com/hypothetical-andrei/wmda-2025/blob/main/c6/example6.py)

---
### 8: Aplicații în NLP
- **Analiza sentimentelor**:
  * Clasificarea polarității textului folosind modele LSTM sau Transformer
- **Sumarizarea textului**:
  * Rezumare extractivă vs. abstractivă
  * Modelele de rezumare folosesc frecvent encoderi cu atenție/Transformere
- **Chatboți**:
  * Arhitecturi secvență-la-secvență + atenție sau bazate pe Transformere
  * Utilizarea modelelor lingvistice pre-antrenate pentru dialog
  * [Exemplu: Utilizează un articol scurt pentru a demonstra cum un model de rezumare abstractivă (precum unul bazat pe BERT) condensează textul](https://github.com/hypothetical-andrei/wmda-2025/blob/main/c6/example7.py)
  * [Exemplu: Rezumarea textului de la zero](https://github.com/hypothetical-andrei/wmda-2025/blob/main/c6/example8.py)

---

### 9: Rezumat & Întrebări
- **Idei Principale**:
  * Reprezentările vectoriale captează semnificația semantică
  * Variantele RNN gestionează dependențele secvențiale
  * Atenția + Transformerele au revoluționat NLP-ul
  * Modelele pre-antrenate (BERT, GPT) ridică performanța la un nou nivel
- **Pașii Următori**:
  * Experimentează cu modele pre-antrenate pe sarcini reale
  * Explorează fine-tuning pentru date specifice unui domeniu
- **Întrebări & Răspunsuri**: Sesiune deschisă de întrebări și discuții
