## **Exercițiul 1 (10 minute): Similaritate Semantică cu Reprezentări Vectoriale**
**Obiectiv**: Înțelegerea modului în care vectorii de cuvinte surprind semnificația folosind operații vectoriale.
1. **Definirea vectorilor de cuvinte**
   * Folosește vectori 3D creați manual pentru `king`, `queen`, `man` și `woman`.
2. **Măsurarea similarității**
   * Calculează similaritățile cosinice între fiecare pereche.
3. **Analogie**
   * Arată `king - man + woman ≈ queen`.
**Puncte cheie**:
* Vectorii encodează relații precum genul sau regalitatea.
* Aritmetica pe embeddings evidențiază structura latentă.

---

## **Exercițiul 2 (10 minute): Antrenarea Word2Vec pe un corpus simplu**
**Obiectiv**: Antrenarea unui model Word2Vec simplu pentru a învăța relații de bază între cuvinte.
1. **Pregătirea setului de date**
   * Folosește un set scurt de propoziții scrise manual.
2. **Antrenarea modelului**
   * Utilizează `gensim.Word2Vec` cu `window=2`, `vector_size=10`.
3. **Explorarea vectorilor**
   * Afișează embeddings, găsește cele mai similare cuvinte.
**Puncte cheie**:
* Word2Vec învață din ferestre de context.
* Chiar și seturi de date mici pot evidenția relații utile.

---

## **Exercițiul 3 (15 minute): Clasificarea sentimentelor cu LSTM**
**Obiectiv**: Antrenarea unui LSTM pentru clasificarea sentimentului textului.
1. **Setul de Date**
   * Utilizează câteva propoziții etichetate manual (pozitiv/negativ).
2. **Modelul**
   * `Embedding` → `LSTM` → `Dense(sigmoid)` folosind Keras.
3. **Antrenare și Evaluare**
   * Utilizează binary crossentropy, urmărește acuratețea.
**Puncte cheie**:
* LSTM-urile procesează textul secvențial.
* Pot surprinde modele în expresii cu încărcătură emoțională.

---

## **Exercițiul 4 (10 minute): Vizualizarea mecanismului de atenție**
**Obiectiv**: Simularea ponderilor de atenție și arătarea modului în care se concentrează pe cuvintele importante.
1. **Stări encoder simulate**
   * Generează stări ascunse aleatorii pentru o propoziție de 5 cuvinte.
2. **Implementarea atenției**
   * Calculează ponderile de atenție folosind un mic modul de atenție.
3. **Vizualizare**
   * Afișează ponderile de atenție sub formă de diagramă bară.
**Puncte cheie**:
* Scorurile de atenție pot fi interpretate vizual.
* Arată „focalizarea” modelului într-o propoziție.

---

## **Exercițiul 5 (10 minute): Harta auto-atenției pentru o propoziție**
**Obiectiv**: Ilustrarea modului în care fiecare cuvânt dintr-o propoziție „se uită” la celelalte cuvinte.
1. **Simularea auto-atenției**
   * Creează scoruri de atenție aleatorii pentru un input de 5 tokenuri.
2. **Vizualizare cu `matplotlib`**
   * Creează o hartă de căldură a ponderilor de atenție.
3. **Etichetarea cuvintelor**
   * Folosește o propoziție simplă, de tipul `„The cat sat on mat”`.
**Puncte cheie**:
* Auto-atenția alimentează Transformerele.
* Fiecare token vede întregul context.

---

## **Exercițiul 6 (10 minute): BERT vs. GPT pentru predicția cuvântului următor dintr-o secvență**
**Obiectiv**: Compararea predicției de cuvinte mascate (BERT) cu continuarea unui prompt (GPT).
1. **Utilizarea unui pipeline HuggingFace**
   * `fill-mask` pentru BERT, `text-generation` pentru GPT.
2. **Rularea exemplului**
   * BERT: `"Capitala Franței este [MASK]."`
   * GPT: `"A fost odată ca niciodată,"`
3. **Discutarea rezultatelor**
   * Ce generează fiecare model?
**Puncte cheie**:
* BERT folosește context bidirecțional.
* GPT generează text secvențial.

---

## **Exercițiul 7 (10 minute): Sumarizarea textului cu BART**
**Obiectiv**: Rezumarea unui paragraf scurt folosind un transformer pre-antrenat.
1. **Input**
   * Folosește câteva propoziții care descriu un eveniment sau o știre.
2. **Generarea rezumatului**
   * Utilizează `facebook/bart-large-cnn` cu `pipeline("summarization")`.
3. **Comparare**
   * Afișează versiunile originale și rezumate.
**Puncte cheie**:
* Modelele pre-antrenate precum BART pot parafraza și comprima.
* Rezumarea este un exemplu de generare secvență-la-secvență.

---
