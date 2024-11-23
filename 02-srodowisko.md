# 2. Instalacja i podstawowa konfiguracja

Wykorzystanie Scikit-learn w projektach związanych z uczeniem maszynowym wymaga poprawnej instalacji biblioteki oraz przygotowania środowiska programistycznego. W tym rozdziale omówimy krok po kroku, jak zainstalować Scikit-learn, skonfigurować środowisko pracy oraz zaimportować kluczowe moduły niezbędne do rozpoczęcia pracy z biblioteką.


## **2.1. Instalacja Scikit-learn**

### **Instalacja za pomocą `pip`**

`pip` to najpopularniejsze narzędzie do zarządzania pakietami w Pythonie. Aby zainstalować Scikit-learn, wystarczy wykonać jedną prostą komendę w terminalu:

```bash
pip install scikit-learn
```

Ta komenda automatycznie zainstaluje najnowszą wersję Scikit-learn oraz wszystkie jej zależności, takie jak:
- **NumPy**: do obsługi macierzy i operacji matematycznych.
- **SciPy**: do zaawansowanych funkcji statystycznych i optymalizacyjnych.
- **joblib**: do równoległego przetwarzania danych.

Aby upewnić się, że instalacja przebiegła pomyślnie, można sprawdzić wersję zainstalowanej biblioteki za pomocą:

```python
import sklearn
print(sklearn.__version__)
```

Jeśli w systemie jest zainstalowana konkretna wersja Pythona i chcemy użyć `pip` dla tej wersji, warto wywołać komendę precyzującą wersję:

```bash
python3 -m pip install scikit-learn
```

### **Instalacja za pomocą `conda`**

Dla użytkowników Anacondy, alternatywą dla `pip` jest menedżer pakietów `conda`. Jest to szczególnie przydatne w środowiskach, gdzie liczy się zarządzanie zależnościami i kompatybilnością między pakietami. Aby zainstalować Scikit-learn za pomocą `conda`, należy użyć następującej komendy:

```bash
conda install scikit-learn
```

`conda` automatycznie zainstaluje wersję Scikit-learn kompatybilną z aktualną wersją Pythona i innymi pakietami, co zmniejsza ryzyko konfliktów zależności.

---

## **2.2. Przygotowanie środowiska programistycznego**

Zanim przystąpimy do pracy ze Scikit-learn, warto przygotować odpowiednie środowisko programistyczne. Zalecane jest korzystanie z wirtualnych środowisk, aby uniknąć konfliktów między różnymi projektami i wersjami pakietów.

### **Tworzenie wirtualnego środowiska z `venv`**

`venv` to wbudowany moduł Pythona, który umożliwia tworzenie izolowanych środowisk. Aby stworzyć nowe środowisko:

1. Utwórz nowe środowisko:
   ```bash
   python3 -m venv my_env
   ```

2. Aktywuj środowisko:
   - W systemie Linux/macOS:
     ```bash
     source my_env/bin/activate
     ```
   - W systemie Windows:
     ```bash
     my_env\Scripts\activate
     ```

3. Zainstaluj Scikit-learn w aktywowanym środowisku:
   ```bash
   pip install scikit-learn
   ```

4. Aby wyłączyć środowisko, użyj komendy:
   ```bash
   deactivate
   ```

### **Tworzenie wirtualnego środowiska z `conda`**

Użytkownicy Anacondy mogą stworzyć nowe środowisko za pomocą `conda`:

1. Utwórz środowisko o nazwie `ml_env` z określoną wersją Pythona:
   ```bash
   conda create --name ml_env python=3.12
   ```

2. Aktywuj środowisko:
   ```bash
   conda activate ml_env
   ```

3. Zainstaluj Scikit-learn:
   ```bash
   conda install scikit-learn
   ```

4. Aby wyłączyć środowisko, użyj komendy:
   ```bash
   conda deactivate
   ```

### **Wybór środowiska IDE**

Do pracy z Scikit-learn najlepiej sprawdzają się interaktywne środowiska programistyczne (IDE) lub edytory kodu wspierające analizę danych. Oto najpopularniejsze opcje:

- **Jupyter Notebook:** Doskonałe środowisko do eksperymentowania z danymi i wizualizacji wyników. Aby zainstalować Jupyter w środowisku:
  ```bash
  pip install notebook
  jupyter notebook
  ```

- **VS Code:** Wszechstronny edytor kodu, który po instalacji rozszerzenia Python doskonale nadaje się do projektów związanych z uczeniem maszynowym.

- **PyCharm:** Popularne IDE dla Pythona, wspierające prace z wirtualnymi środowiskami i analizą kodu.

---

## **2.3. Importowanie kluczowych modułów**

Po poprawnej instalacji Scikit-learn i konfiguracji środowiska możemy zaimportować niezbędne moduły. Warto znać strukturę biblioteki, aby w łatwy sposób korzystać z jej funkcji.

### **Podstawowe importy**

Do pracy z klasyfikacją, regresją czy przetwarzaniem danych, najczęściej używa się następujących modułów:

```python
# Import klasyfikatorów
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Import modeli regresji
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Import narzędzi do przetwarzania danych
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Import narzędzi do walidacji modeli
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
```

### **Przykładowy przepływ pracy z Scikit-learn**

Poniżej przedstawiono prosty przykład użycia Scikit-learn do rozwiązania problemu klasyfikacji na podstawie wbudowanego zestawu danych:

1. Zaimportuj zestaw danych Iris:
   ```python
   from sklearn.datasets import load_iris
   ```

2. Podziel dane na zestawy treningowy i testowy:
   ```python
   from sklearn.model_selection import train_test_split

   iris = load_iris()
   X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
   ```

3. Wybierz i zaimportuj model klasyfikacji, np. drzewa decyzyjne:
   ```python
   from sklearn.tree import DecisionTreeClassifier
   model = DecisionTreeClassifier()
   ```

4. Wytrenuj model na danych treningowych:
   ```python
   model.fit(X_train, y_train)
   ```

5. Przeprowadź predykcję i oceń model:
   ```python
   predictions = model.predict(X_test)
   from sklearn.metrics import accuracy_score
   print(f"Dokładność modelu: {accuracy_score(y_test, predictions):.2f}")
   ```

### **Integracja z innymi bibliotekami**

Praca ze Scikit-learn często wymaga użycia innych bibliotek, takich jak:
- **NumPy:** do manipulacji danymi:
  ```python
  import numpy as np
  array = np.array([[1, 2], [3, 4]])
  ```
- **Pandas:** do pracy z tabelarycznymi danymi:
  ```python
  import pandas as pd
  df = pd.DataFrame(data=array, columns=["Feature1", "Feature2"])
  ```
- **Matplotlib i Seaborn:** do wizualizacji danych i wyników:
  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns

  sns.pairplot(df)
  plt.show()
  ```


### Podsumowanie

Proces instalacji i konfiguracji Scikit-learn jest prosty i szybki, a biblioteka została zaprojektowana z myślą o intuicyjnym użytkowaniu. Wykorzystanie `pip` lub `conda` do instalacji, przygotowanie wirtualnego środowiska oraz zaimportowanie kluczowych modułów to podstawowe kroki pozwalające rozpocząć pracę z uczeniem maszynowym. Dzięki bogatej integracji z innymi bibliotekami Python, Scikit-learn pozwala w łatwy sposób tworzyć kompleksowe przepływy analityczne. Kolejne rozdziały pokażą, jak wykorzystać te narzędzia w praktyce.

---
