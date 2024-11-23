# 3. Struktura Scikit-learn

Scikit-learn to potężna biblioteka do uczenia maszynowego, która została zaprojektowana z myślą o prostocie i spójności. Jego struktura opiera się na kilku kluczowych komponentach, które umożliwiają efektywne budowanie i ocenę modeli uczenia maszynowego. W tym rozdziale omówimy główne elementy biblioteki oraz zasady korzystania z jej API.


## **3.1. Główne komponenty**

Jednym z największych atutów Scikit-learn jest modularna konstrukcja, w której wszystkie elementy, takie jak estymatory, procesory i narzędzia do walidacji, są zintegrowane w spójny sposób. Dzięki temu użytkownicy mogą łatwo zrozumieć i wykorzystać funkcjonalności biblioteki.

### **3.1.1. Estymatory**

Estymatory to centralny element Scikit-learn. Są to obiekty reprezentujące modele uczenia maszynowego (np. regresję liniową, drzewa decyzyjne) lub narzędzia do przetwarzania danych (np. skalowanie, imputacja). Estymatory działają na zasadzie dwóch podstawowych metod: 
- **`fit`**: służy do uczenia modelu na danych treningowych.
- **`predict`**: umożliwia generowanie przewidywań na podstawie wytrenowanego modelu.
- **`transform`**: używane głównie w procesorach do przekształcania danych.

#### Przykład: Klasyfikator
```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Załadowanie danych
data = load_iris()
X, y = data.data, data.target

# Utworzenie modelu i jego trenowanie
model = LogisticRegression()
model.fit(X, y)

# Przewidywanie na podstawie modelu
predictions = model.predict(X)
```

#### Przykład: Procesor
```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Dane
X = np.array([[1, 2], [3, 4], [5, 6]])

# Skalowanie danych
scaler = StandardScaler()
scaler.fit(X)  # Dopasowanie do danych
X_scaled = scaler.transform(X)  # Przekształcenie danych
```

Każdy estymator w Scikit-learn działa w spójny sposób, co znacząco ułatwia przechodzenie między różnymi algorytmami lub narzędziami.

---

### **3.1.2. Procesory**

Procesory to kluczowe narzędzia wspierające przygotowanie danych i automatyzację procesu modelowania. W Scikit-learn procesory obejmują:
- **Preprocessing (przetwarzanie danych):** narzędzia do normalizacji, imputacji brakujących wartości, kodowania zmiennych kategorycznych itp.
- **Pipeline:** mechanizm umożliwiający łączenie różnych etapów przetwarzania i modelowania w jednym, zautomatyzowanym przepływie.

#### Preprocessing

Preprocessing jest często pierwszym krokiem w procesie budowy modelu. Dane rzeczywiste często wymagają przekształceń, aby mogły być poprawnie przetworzone przez algorytmy uczenia maszynowego.

##### Skalowanie
Normalizacja lub standaryzacja danych jest kluczowa dla wielu algorytmów:
```python
from sklearn.preprocessing import MinMaxScaler

# Normalizacja danych
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
```

##### Kodowanie zmiennych kategorycznych
W przypadku zmiennych nienumerycznych konieczne jest ich zakodowanie:
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform([['cat'], ['dog'], ['mouse']]).toarray()
```

#### Pipeline

Pipeline umożliwia połączenie wielu etapów przetwarzania danych i modelowania w jeden obiekt. Jest to szczególnie przydatne w bardziej złożonych przepływach pracy.

Przykład użycia Pipeline:
```python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Połączenie skalowania i modelu SVM w jeden pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# Trenowanie pipeline
pipeline.fit(X, y)

# Przewidywanie przy użyciu pipeline
predictions = pipeline.predict(X)
```

Pipeline automatyzuje przetwarzanie danych i pozwala uniknąć powtarzania tych samych kroków przy każdym uruchomieniu kodu.

---

## **3.2. Zasady korzystania z API biblioteki**

Scikit-learn został zaprojektowany z myślą o jednolitym i spójnym interfejsie użytkownika. Dzięki temu praca z różnymi algorytmami i narzędziami jest intuicyjna, co pozwala użytkownikom szybko zmieniać i dostosowywać swoje rozwiązania.

### **3.2.1. Jednolity interfejs**

Każdy estymator w Scikit-learn stosuje te same metody do treningu, przewidywania i przekształcania danych. Kluczowe metody to:
- **`fit(X, y)`**: dopasowanie modelu lub procesora do danych treningowych.
- **`predict(X)`**: przewidywanie na podstawie wytrenowanego modelu.
- **`transform(X)`**: przekształcenie danych, np. normalizacja lub redukcja wymiarowości.

Przykład: użycie różnych algorytmów klasyfikacji w tym samym przepływie:
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Dane
X = [[0, 0], [1, 1]]
y = [0, 1]

# Trenowanie i przewidywanie z użyciem różnych modeli
models = [DecisionTreeClassifier(), RandomForestClassifier()]
for model in models:
    model.fit(X, y)
    print(model.predict([[0.5, 0.5]]))
```

### **3.2.2. Klasyfikacja, regresja, klastrowanie**

Scikit-learn obsługuje trzy główne kategorie algorytmów uczenia maszynowego:

#### Klasyfikacja
Klasyfikacja polega na przypisywaniu przykładów do jednej z wcześniej zdefiniowanych klas. Przykłady algorytmów:
- **Logistic Regression**: dla problemów liniowych.
- **Support Vector Machines (SVM)**: dla danych o wysokiej liczbie wymiarów.
- **Random Forest**: dla złożonych danych.

Przykład:
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)
predictions = model.predict(X)
```

#### Regresja
Regresja przewiduje wartości ciągłe, takie jak ceny lub temperatury. Przykłady algorytmów:
- **Linear Regression**: dla relacji liniowych.
- **Ridge i Lasso Regression**: dla regularizacji.
- **Random Forest Regressor**: dla danych nieliniowych.

Przykład:
```python
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X, y)
predictions = regressor.predict(X)
```

#### Klastrowanie
Klastrowanie to metoda grupowania danych w klastry bez potrzeby wcześniejszego oznaczania przykładów. Przykłady algorytmów:
- **K-means**: do dzielenia danych na zdefiniowaną liczbę grup.
- **DBSCAN**: do klastrowania danych o dowolnym kształcie.
- **Hierarchiczne klastrowanie**: do tworzenia dendrogramów.

Przykład:
```python
from sklearn.cluster import KMeans

clusterer = KMeans(n_clusters=2)
clusterer.fit(X)
labels = clusterer.predict(X)
```


### Podsumowanie

Struktura Scikit-learn opiera się na kilku kluczowych komponentach: estymatorach, procesorach i spójnym API. Dzięki temu użytkownicy mogą w prosty sposób budować modele uczenia maszynowego, przetwarzać dane i analizować wyniki. Jednolity interfejs sprawia, że praca z różnymi algorytmami jest intuicyjna, a narzędzia takie jak Pipeline automatyzują procesy przetwarzania i modelowania. Scikit-learn to narzędzie zarówno dla początkujących, jak i zaawansowanych użytkowników, pozwalające na efektywne rozwiązywanie problemów analizy danych i uczenia maszynowego.

---
