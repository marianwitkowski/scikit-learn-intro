# **5. Przetwarzanie danych**

Przetwarzanie danych jest kluczowym krokiem w procesie budowania modeli uczenia maszynowego. Dane rzeczywiste są często niestandardowe, niekompletne lub nienormalizowane, co wymaga ich odpowiedniego przygotowania przed zastosowaniem algorytmów uczenia maszynowego. W tym rozdziale omówimy techniki wstępnego przetwarzania danych, obsługę brakujących wartości oraz tworzenie potoków (ang. pipelines) w Scikit-learn. Wszystkie przedstawione narzędzia mają na celu uproszczenie i automatyzację procesu przygotowania danych.


## **5.1. Wstępne przetwarzanie**

Wstępne przetwarzanie danych jest podstawą skutecznego modelowania. Dane wejściowe muszą być w odpowiednim formacie i dobrze przygotowane, aby algorytmy uczenia maszynowego mogły działać poprawnie. Do najważniejszych kroków należą: skalowanie danych numerycznych oraz kodowanie zmiennych kategorycznych.

---

### **Skalowanie**

Skalowanie danych jest istotne, ponieważ wiele algorytmów uczenia maszynowego (np. SVM, kNN, regresja liniowa) jest wrażliwych na różnice w skalach cech. W Scikit-learn dostępne są różne podejścia do skalowania:

#### **StandardScaler**
Metoda ta skaluje dane, tak aby miały średnią równą 0 i odchylenie standardowe równe 1. Jest to szczególnie przydatne w algorytmach zakładających normalność danych.

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Przykładowe dane
X = np.array([[1, 2], [2, 4], [3, 6]])

# Skalowanie
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Zeskalowane dane: \n{X_scaled}")
```

#### **MinMaxScaler**
MinMaxScaler skaluje dane w taki sposób, aby wartości mieściły się w zdefiniowanym przedziale (domyślnie od 0 do 1). Jest to szczególnie przydatne w przypadkach, gdy chcemy zachować proporcje między cechami.

```python
from sklearn.preprocessing import MinMaxScaler

# Skalowanie do zakresu 0-1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print(f"Zeskalowane dane (0-1): \n{X_scaled}")
```

---

### **Kodowanie zmiennych kategorycznych**

W przypadku zmiennych kategorycznych, algorytmy uczenia maszynowego wymagają reprezentacji liczbowej. Najczęściej stosowane metody to:

#### **OneHotEncoder**
OneHotEncoder zamienia zmienne kategoryczne na wektory binarne. Dla każdej kategorii tworzy nową kolumnę, a wiersze są oznaczone wartością 1, jeśli dana próbka należy do tej kategorii, lub 0 w przeciwnym razie.

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Dane kategoryczne
categories = np.array([['cat'], ['dog'], ['mouse']])

# Kodowanie OneHot
encoder = OneHotEncoder()
encoded = encoder.fit_transform(categories).toarray()
print(f"Zakodowane dane: \n{encoded}")
```

**Zalety OneHotEncoder:**
- Umożliwia bezstronną reprezentację zmiennych kategorycznych.
- Zapobiega wprowadzeniu fałszywych relacji porządkowych między kategoriami.

---

## **5.2. Obsługa brakujących wartości**

Dane rzeczywiste często zawierają brakujące wartości, które mogą wpłynąć na wyniki modelu. Scikit-learn oferuje narzędzia do imputacji, czyli uzupełniania brakujących wartości w danych.

---

### **SimpleImputer**

SimpleImputer to prosty, ale skuteczny sposób na uzupełnienie brakujących wartości. Możemy wypełnić braki wartością średnią, medianą, najczęściej występującą wartością lub dowolną inną, którą zdefiniujemy.

#### Przykład:
```python
from sklearn.impute import SimpleImputer
import numpy as np

# Dane z brakującymi wartościami
X = np.array([[1, 2, np.nan], [3, np.nan, 6], [7, 8, 9]])

# Imputacja średnią
imputer = SimpleImputer(strategy='mean')
X_filled = imputer.fit_transform(X)
print(f"Dane po imputacji: \n{X_filled}")
```

---

### **IterativeImputer**

IterativeImputer wykorzystuje bardziej zaawansowane podejście, w którym brakujące wartości są iteracyjnie przewidywane na podstawie innych cech. Algorytm ten stosuje model regresji do estymacji brakujących danych, co pozwala na uzyskanie bardziej dokładnych wyników.

#### Przykład:
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Imputacja iteracyjna
iter_imputer = IterativeImputer()
X_iterative = iter_imputer.fit_transform(X)
print(f"Dane po imputacji iteracyjnej: \n{X_iterative}")
```

**Porównanie SimpleImputer i IterativeImputer:**
- **SimpleImputer:** Szybszy, mniej złożony, ale mniej dokładny.
- **IterativeImputer:** Bardziej zaawansowany, szczególnie przydatny w przypadku silnych korelacji między cechami.

---

## **5.3. Pipeline’y**

Pipeline to potężne narzędzie w Scikit-learn, które umożliwia łączenie różnych etapów przetwarzania danych i modelowania w jeden zautomatyzowany przepływ. Dzięki pipeline’om można uniknąć błędów związanych z ręcznym przetwarzaniem danych i poprawić czytelność kodu.

---

### **Automatyzacja przetwarzania i modelowania**

Pipeline pozwala zdefiniować sekwencję kroków przetwarzania danych oraz trenowania modelu, które są automatycznie wykonywane w odpowiedniej kolejności. Przykładowy pipeline może obejmować:
- Skalowanie danych,
- Kodowanie zmiennych kategorycznych,
- Trenowanie modelu.

#### Przykład:
```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Tworzenie pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Dane
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])

# Trenowanie modelu
pipeline.fit(X, y)

# Przewidywanie
predictions = pipeline.predict(X)
print(f"Przewidywania: {predictions}")
```

**Korzyści z użycia Pipeline:**
- Automatyzacja przetwarzania danych i modelowania.
- Możliwość łatwej modyfikacji poszczególnych kroków.
- Zmniejszenie ryzyka błędów w przepływie danych.

---

### **Tworzenie własnych potoków**

Pipeline można dostosować do własnych potrzeb, definiując niestandardowe kroki. Aby to zrobić, wystarczy zaimplementować klasę z metodami `fit` i `transform`.

#### Przykład niestandardowego kroku:
```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Przykładowa niestandardowa transformacja
        return X + 1

# Użycie niestandardowego transformera w pipeline
custom_pipeline = Pipeline([
    ('custom_transformer', CustomTransformer()),
    ('scaler', StandardScaler())
])

# Transformacja danych
X_custom = custom_pipeline.fit_transform(X)
print(f"Zmienione dane: \n{X_custom}")
```

### Podsumowanie

Przetwarzanie danych jest kluczowym elementem każdego projektu uczenia maszynowego. Scikit-learn oferuje szeroką gamę narzędzi do wstępnego przetwarzania danych, obsługi brakujących wartości oraz automatyzacji procesów za pomocą pipeline’ów. Dzięki takim funkcjom użytkownicy mogą skupić się na tworzeniu i optymalizacji modeli, mając pewność, że dane wejściowe są odpowiednio przygotowane. Każdy krok w przetwarzaniu danych, od skalowania po imputację, można łatwo zintegrować w jednym spójnym przepływie, co czyni Scikit-learn nieocenionym narzędziem w analizie danych.

---
