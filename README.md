# **Scikit-learn: Przewodnik po najważniejszych możliwościach i zastosowaniach**

<i>Zanim wszystko zaczniesz wrzucasz do prompta ChatGPT, warto poznać jak to drzewiej bywało, i że znająć się na analizie danych oraz uczeniu maszynowym też jesteś w stanie tworzy niezłe modele....</i>


#### &copy; 2024 - Marian Witkowski marian.witkowski[at]gmail.com

### **1. <a href='01-wprowadzenie.md'>Wprowadzenie</a>**
   - **1.1. Czym jest Scikit-learn?**
     - Krótka charakterystyka: popularna biblioteka Python do uczenia maszynowego.
     - Kluczowe cechy: prostota, wszechstronność, efektywność.
   - **1.2. Dlaczego warto korzystać z Scikit-learn?**
     - Wspierane algorytmy i narzędzia.
     - Integracja z innymi bibliotekami (NumPy, Pandas, Matplotlib).

### **2. <a href='02-srodowisko.md'>Instalacja i podstawowa konfiguracja</a>**
   - Instrukcja instalacji przez `pip` i `conda`.
   - Przygotowanie środowiska programistycznego.
   - Importowanie kluczowych modułów.

### **3. <a href='03-struktura.md'>Struktura Scikit-learn</a>**
   - **3.1. Główne komponenty:**
     - Estymatory (`fit`, `predict`, `transform`).
     - Procesory (`pipeline`, `preprocessing`).
   - **3.2. Zasady korzystania z API biblioteki:**
     - Jednolity interfejs.
     - Klasyfikacja, regresja, klastrowanie.

### **4. <a href='04-zastosowanie.md'>Przykładowe zastosowania Scikit-learn</a>**
   - **4.1. Klasyfikacja**
     - Przykład: rozpoznawanie irysów na podstawie zestawu danych Iris.
     - Algorytmy: drzewa decyzyjne, SVM, kNN.
   - **4.2. Regresja**
     - Przykład: przewidywanie cen nieruchomości.
     - Algorytmy: regresja liniowa, Lasso, Ridge.
   - **4.3. Klastrowanie**
     - Przykład: segmentacja klientów.
     - Algorytmy: K-means, hierarchiczne klastrowanie.
   - **4.4. Redukcja wymiarowości**
     - Przykład: analiza danych obrazowych.
     - PCA, t-SNE.
   - **4.5. Wykrywanie anomalii**
     - Przykład: identyfikacja oszustw finansowych.
     - Algorytmy: Isolation Forest, One-class SVM.

### **5. <a href='05-przetwarzanie.md'>Przetwarzanie danych</a>**
   - **5.1. Wstępne przetwarzanie**
     - Skalowanie (StandardScaler, MinMaxScaler).
     - Kodowanie zmiennych kategorycznych (OneHotEncoder).
   - **5.2. Obsługa brakujących wartości**
     - SimpleImputer, IterativeImputer.
   - **5.3. Pipeline’y**
     - Automatyzacja przetwarzania i modelowania.
     - Tworzenie własnych potoków.

### **6. <a href='06-walidacja.md'>Walidacja modeli i dobór hiperparametrów</a>**
   - **6.1. Walidacja krzyżowa**
     - Funkcja `cross_val_score`.
   - **6.2. Dobór hiperparametrów**
     - GridSearchCV i RandomizedSearchCV.
   - **6.3. Metryki oceny modelu**
     - Klasyfikacja: Accuracy, F1-score, ROC AUC.
     - Regresja: MSE, MAE, R^2.

### **7. <a href='07-integracja.md'>Integracja z innymi narzędziami</a>**
   - Wykorzystanie Pandas do analizy danych.
   - Wizualizacja wyników za pomocą Matplotlib i Seaborn.
   - Eksportowanie modeli do formatu Pickle lub ONNX.

### **8. <a href='08-praktyki.md'>Wyzwania i najlepsze praktyki</a>**
   - Unikanie przeuczenia.
   - Interpretowalność modeli.
   - Dokumentacja i źródła dodatkowe (przewodniki, tutoriale).

---
