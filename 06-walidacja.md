# **6. Walidacja modeli i dobór hiperparametrów**

Proces uczenia maszynowego nie kończy się na trenowaniu modelu. Aby stworzyć skuteczny i niezawodny system, należy odpowiednio ocenić jego wydajność, uniknąć przeuczenia oraz dostroić hiperparametry. W tym rozdziale szczegółowo omówimy walidację modeli, dobór hiperparametrów oraz metryki oceny, które są kluczowe w tworzeniu optymalnych modeli uczenia maszynowego.


## **6.1. Walidacja krzyżowa**

Walidacja krzyżowa to technika oceny wydajności modelu, która pozwala zmniejszyć ryzyko przeuczenia i uzyskać bardziej realistyczny obraz jego skuteczności. Zamiast używać jednej, stałej części danych do walidacji, walidacja krzyżowa dzieli dane na kilka podzbiorów (tzw. "folds") i wielokrotnie trenuje model na różnych kombinacjach danych treningowych i testowych.

---

### **K-fold cross-validation**

Najpopularniejszą techniką walidacji krzyżowej jest **K-fold cross-validation**. W tej metodzie dane są dzielone na \(k\) równych podzbiorów. Model jest trenowany \(k\) razy, za każdym razem używając innego podzbioru jako danych testowych, a pozostałych jako danych treningowych. Wynik oceny to średnia wartość metryki z \(k\) iteracji.

#### Zalety K-fold cross-validation:
1. **Wiarygodna ocena:** Wykorzystuje całe dane do oceny modelu.
2. **Mniejsze ryzyko przeuczenia:** Lepsza generalizacja wyników.
3. **Elastyczność:** Można dostosować liczbę podzbiorów (\(k\)).

#### Przykład z `cross_val_score`:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Załadowanie danych
iris = load_iris()
X, y = iris.data, iris.target

# Model
model = DecisionTreeClassifier()

# Walidacja krzyżowa
scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
print(f"Wyniki walidacji krzyżowej: {scores}")
print(f"Średnia dokładność: {scores.mean():.2f}")
```

---

### **Leave-One-Out Cross-Validation (LOOCV)**

LOOCV to skrajny przypadek walidacji krzyżowej, w którym liczba podzbiorów \(k\) jest równa liczbie próbek w zbiorze danych. Każda próbka jest używana jako dane testowe dokładnie raz.

#### Przykład:

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
print(f"Średnia dokładność LOOCV: {scores.mean():.2f}")
```

---

### **Stratyfikowana walidacja krzyżowa**

W przypadku problemów klasyfikacyjnych z niezrównoważonymi klasami warto użyć **stratyfikowanej walidacji krzyżowej**. Zapewnia ona, że proporcje klas są zachowane w każdym podzbiorze.

#### Przykład:

```python
from sklearn.model_selection import StratifiedKFold

stratified_cv = StratifiedKFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=stratified_cv)
print(f"Wyniki stratyfikowanej walidacji: {scores}")
```

---

## **6.2. Dobór hiperparametrów**

Hiperparametry to wartości konfiguracyjne, które muszą być ustawione przed rozpoczęciem treningu modelu (np. liczba drzew w Random Forest, wartość \(C\) w SVM). Ich właściwy dobór ma kluczowe znaczenie dla osiągnięcia optymalnej wydajności modelu. Scikit-learn oferuje dwa główne narzędzia do tego celu: `GridSearchCV` i `RandomizedSearchCV`.

---

### **GridSearchCV**

`GridSearchCV` wykonuje przeszukiwanie siatki hiperparametrów, testując wszystkie możliwe kombinacje w określonym zakresie. Jest to metoda wyczerpująca, ale czasochłonna, szczególnie dla dużych zbiorów danych lub dużej liczby hiperparametrów.

#### Przykład:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Model
model = RandomForestClassifier()

# Parametry do przeszukania
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# GridSearch
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)

# Najlepsze parametry
print(f"Najlepsze parametry: {grid_search.best_params_}")
print(f"Najlepszy wynik: {grid_search.best_score_:.2f}")
```

---

### **RandomizedSearchCV**

`RandomizedSearchCV` przeszukuje losowy podzbiór możliwych kombinacji hiperparametrów. Jest szybszy niż GridSearch, szczególnie przy dużych przestrzeniach parametrów, i pozwala znaleźć dobre wartości bez pełnego przeszukiwania siatki.

#### Przykład:

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Model
model = RandomForestClassifier()

# Parametry do losowego przeszukania
param_dist = {
    'n_estimators': np.arange(10, 200, 10),
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# RandomizedSearch
random_search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=5, random_state=42)
random_search.fit(X, y)

# Najlepsze parametry
print(f"Najlepsze parametry: {random_search.best_params_}")
print(f"Najlepszy wynik: {random_search.best_score_:.2f}")
```

#### Porównanie GridSearchCV i RandomizedSearchCV:
| **Cecha**           | **GridSearchCV**                   | **RandomizedSearchCV**             |
|----------------------|------------------------------------|-------------------------------------|
| **Dokładność**       | Testuje wszystkie kombinacje       | Losowe podzbiory                   |
| **Szybkość**         | Czasochłonny                      | Szybszy                            |
| **Skalowalność**     | Trudności przy dużych przestrzeniach | Bardziej wydajny przy dużych zestawach parametrów |

---

## **6.3. Metryki oceny modelu**

Metryki oceny pozwalają zmierzyć jakość działania modelu na podstawie wyników predykcji. Wybór odpowiedniej metryki zależy od typu problemu (klasyfikacja lub regresja) i wymagań analizy.

---

### **Metryki dla klasyfikacji**

1. **Accuracy (dokładność):**
   Odsetek poprawnych przewidywań względem wszystkich obserwacji. Jest to podstawowa metryka, ale nie sprawdza się w przypadku niezrównoważonych danych.

   ```python
   from sklearn.metrics import accuracy_score

   # Przewidywania
   y_pred = model.predict(X)
   print(f"Accuracy: {accuracy_score(y, y_pred):.2f}")
   ```

2. **F1-score:**
   Średnia harmoniczna precyzji i czułości, która jest szczególnie użyteczna w przypadku niezrównoważonych klas.

   ```python
   from sklearn.metrics import f1_score

   print(f"F1-score: {f1_score(y, y_pred, average='weighted'):.2f}")
   ```

3. **ROC AUC:**
   Pole pod krzywą ROC (Receiver Operating Characteristic) mierzy zdolność modelu do rozróżniania między klasami. Wartość 1 oznacza idealny model, 0.5 - model losowy.

   ```python
   from sklearn.metrics import roc_auc_score

   # Wartości przewidywane prawdopodobieństwa
   y_prob = model.predict_proba(X)[:, 1]
   print(f"ROC AUC: {roc_auc_score(y, y_prob):.2f}")
   ```

---

### **Metryki dla regresji**

1. **Mean Squared Error (MSE):**
   Średnia kwadratowa różnica między przewidywaniami a rzeczywistymi wartościami. MSE karze za duże błędy bardziej niż MAE.

   ```python
   from sklearn.metrics import mean_squared_error

   print(f"MSE: {mean_squared_error(y, y_pred):.2f

}")
   ```

2. **Mean Absolute Error (MAE):**
   Średnia bezwzględna różnica między przewidywaniami a rzeczywistymi wartościami. Jest bardziej odporna na wpływ odległych wartości.

   ```python
   from sklearn.metrics import mean_absolute_error

   print(f"MAE: {mean_absolute_error(y, y_pred):.2f}")
   ```

3. **R-squared (R²):**
   Współczynnik determinacji mierzy, jaka część wariancji w danych jest wyjaśniona przez model. Wartość 1 oznacza idealne dopasowanie, a wartość 0 oznacza, że model nie wyjaśnia żadnej wariancji.

   ```python
   from sklearn.metrics import r2_score

   print(f"R²: {r2_score(y, y_pred):.2f}")
   ```


### Podsumowanie

Walidacja modeli i dobór hiperparametrów są kluczowymi krokami w procesie budowy modeli uczenia maszynowego. Walidacja krzyżowa, za pomocą takich narzędzi jak `cross_val_score`, pozwala ocenić wydajność modelu w sposób bardziej reprezentatywny. Dobór hiperparametrów przy użyciu `GridSearchCV` lub `RandomizedSearchCV` umożliwia optymalizację modelu pod kątem wydajności. Wybór odpowiednich metryk oceny, takich jak Accuracy, F1-score czy R², pozwala lepiej zrozumieć jakość działania modelu w zależności od charakteru problemu. Scikit-learn oferuje wszechstronne narzędzia wspierające te procesy, co czyni go nieocenionym narzędziem w pracy z uczeniem maszynowym.

---
