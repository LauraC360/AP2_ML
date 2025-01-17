# AP2_ML
Aplicație Practică 2 de Învățare Automată

Dezvoltat de studenții: Chiriac Laura-Florina & Bindiu Ana-Maria

## Descrierea Proiectului

Acest proiect are ca scop studierea adaptării și adecvării diferiților algoritmi de învățare supervizată pentru a rezolva problema maximizării profitului pentru un proprietar de lanț hotelier. Scopul este de a propune un clasament al celor șase categorii de activități posibile (Nature, Historical, Cultural, Beach, Adventure, Urban) pentru a maximiza profitul (Revenue) și/sau profitul pe cap de vizitator (Revenue/Visitors) într-o țară dată.

## Analiza Setului de Date

În cadrul acestui studiu, am analizat un set de date disponibil la adresa specificată, care descrie informații turistice relevante, incluzând următoarele atribute: **Location**, **Country**, **Category**, **Visitors**, **Rating**, **Revenue**, și **Accommodation_Available**. Scopul principal este de a propune o soluție bazată pe algoritmi de Învățare Automată, care să determine o ierarhie optimă a categoriilor de activități tematice (ex.: *Nature, Historical, Cultural* etc.) ce maximizează profitul (**Revenue**) și/sau profitul per vizitator (**Revenue/Visitors**) pentru o anumită țară.

## Metodologie

1. **Înțelegerea și Preprocesarea Datelor**:
   - Înțelegerea atributelor și a corelațiilor dintre ele.
   - Preprocesarea datelor pentru a filtra după țară, a calcula profitul pe vizitator și a codifica variabilele categorice.

2. **Selecția și Implementarea Algoritmilor**:
   - Implementarea și evaluarea diferiților algoritmi de învățare supervizată (ID3, kNN, AdaBoost).
   - Selectarea celui mai potrivit algoritm pe baza justificării teoretice și experimentale.

3. **Evaluarea Performanței**:
   - Împărțirea setului de date în seturi de antrenament și testare (în raportul 80%-20%).
   - Evaluarea performanței algoritmului selectat folosind metrici relevante și vizualizări (de exemplu, pie charts).

## Rezultate

Rezultatele experimentelor, inclusiv performanța diferiților algoritmi și clasamentul final al categoriilor de activități, sunt documentate într-un raport LaTeX.

## Cum se rulează

1. Se rulează pe rând adaboost.py, id3.py și knn.py pentru a obține rezultatele pentru fiecare algoritm în parte.
2. Se rulează main.py pentru a obține rezultatele finale legate de clasamentul categoriilor de activități.
3. Se rulează main_comparative_analysis.py pentru a obține rezultatele comparative ale performanțelor celor trei algoritmi.

## Fișiere

- `src/main.py`: Script principal pentru preprocesarea datelor și antrenarea pe setul de date.
- `src/knn.py`: Implementarea algoritmului kNN.
- `src/id3.py`: Implementarea algoritmului ID3.
- `src/adaboost.py`: Implementarea algoritmului AdaBoost.
- `src/main_comparative_analysis.py`: Script pentru analiza comparativă a diferiților algoritmi.
- `src/adaboost_results.txt`: Rezultatele modelului AdaBoost pentru sarcina de clasificare.
- `README.md`: Acest fișier.