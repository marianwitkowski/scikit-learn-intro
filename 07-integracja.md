# **7. Integracja z innymi narzędziami**

Scikit-learn, będąc wszechstronną biblioteką uczenia maszynowego, zyskuje na funkcjonalności dzięki doskonałej integracji z innymi narzędziami w ekosystemie Pythona. Narzędzia te, takie jak **Pandas** do analizy danych, **Matplotlib** i **Seaborn** do wizualizacji wyników oraz **Pickle** i **ONNX** do eksportowania modeli, są powszechnie używane przez specjalistów ds. danych. W tym rozdziale szczegółowo omówimy, jak wykorzystać te narzędzia w połączeniu ze Scikit-learn.


## **7.1. Wykorzystanie Pandas do analizy danych**

Pandas to popularna biblioteka do analizy i manipulacji danych, która doskonale współpracuje z Scikit-learn. Dane są zazwyczaj wczytywane i przetwarzane w Pandas w formacie DataFrame, co ułatwia ich analizę, czyszczenie oraz przygotowanie do trenowania modeli uczenia maszynowego.

### **Wczytywanie i eksploracja danych**

Przykład: Wczytanie zestawu danych Iris i podstawowa analiza.

```python
import pandas as pd
from sklearn.datasets import load_iris

# Załadowanie danych Iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Podstawowe informacje o danych
print(df.head())  # Pierwsze 5 wierszy
print(df.info())  # Informacje o typach danych
print(df.describe())  # Statystyki opisowe
```

### **Manipulacja danych**

Pandas umożliwia łatwą manipulację danych, np. filtrowanie, grupowanie czy agregację.

```python
# Filtrowanie danych dla jednego gatunku
filtered_data = df[df['species'] == 0]

# Grupowanie i obliczanie średniej dla każdej cechy
grouped_data = df.groupby('species').mean()
print(grouped_data)
```

### **Przygotowanie danych do Scikit-learn**

Pandas ułatwia konwersję danych do formatu NumPy, który jest wymagany przez większość algorytmów w Scikit-learn.

```python
X = df.iloc[:, :-1].values  # Wszystkie kolumny poza ostatnią
y = df['species'].values  # Kolumna docelowa
```

Pandas i Scikit-learn w połączeniu tworzą potężne narzędzie do analizy danych. Dzięki intuicyjnej manipulacji w Pandas dane można łatwo przygotować do modelowania w Scikit-learn.

---

## **7.2. Wizualizacja wyników za pomocą Matplotlib i Seaborn**

Wizualizacja danych i wyników jest kluczowa dla zrozumienia ich struktury oraz oceny skuteczności modeli. Matplotlib i Seaborn to dwie popularne biblioteki Python, które doskonale integrują się ze Scikit-learn.

### **Matplotlib**

Matplotlib to wszechstronna biblioteka do tworzenia wykresów w Pythonie. Przykłady jej zastosowania w połączeniu z Scikit-learn obejmują wykresy punktowe, histogramy oraz wizualizacje metryk modeli.

#### Przykład: Wykres punktowy dla cech danych Iris
```python
import matplotlib.pyplot as plt

# Wykres punktowy dla dwóch cech
plt.figure(figsize=(8, 6))
for species, group in df.groupby('species'):
    plt.scatter(group.iloc[:, 0], group.iloc[:, 1], label=f'Species {species}')
plt.title('Scatter Plot: Iris Dataset')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.show()
```

#### Przykład: Histogram
```python
# Histogram jednej z cech
plt.hist(df.iloc[:, 0], bins=20, color='blue', alpha=0.7)
plt.title('Histogram: Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()
```

### **Seaborn**

Seaborn to biblioteka do tworzenia bardziej zaawansowanych wizualizacji opartych na Matplotlib. Szczególnie przydatna do analiz wielowymiarowych danych i wizualizacji zależności między cechami.

#### Przykład: Pairplot dla danych Iris
```python
import seaborn as sns

# Pairplot z podziałem na gatunki
sns.pairplot(df, hue='species', palette='husl')
plt.suptitle("Seaborn Pairplot: Iris Dataset", y=1.02)
plt.show()
```

#### Przykład: Heatmapa korelacji między cechami
```python
# Obliczanie macierzy korelacji
correlation_matrix = df.iloc[:, :-1].corr()

# Heatmapa korelacji
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap: Feature Correlations')
plt.show()
```

Matplotlib i Seaborn w połączeniu umożliwiają szczegółowe zrozumienie danych i ocenę wyników modeli Scikit-learn.

---

## **7.3. Eksportowanie modeli do formatu Pickle lub ONNX**

Po wytrenowaniu modelu w Scikit-learn często zachodzi potrzeba jego zapisania, aby można go było później załadować bez konieczności ponownego trenowania. Dwa popularne formaty do tego celu to Pickle i ONNX.

---

### **Pickle**

Pickle to wbudowana w Python biblioteka do serializacji obiektów, w tym modeli Scikit-learn. Jest łatwa w użyciu i szeroko stosowana do lokalnego przechowywania modeli.

#### Przykład: Zapisywanie i ładowanie modelu za pomocą Pickle

```python
import pickle
from sklearn.linear_model import LogisticRegression

# Trening modelu
model = LogisticRegression(max_iter=200)
X = iris.data
y = iris.target
model.fit(X, y)

# Zapis modelu
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Wczytanie modelu
with open('iris_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Testowanie wczytanego modelu
predicted = loaded_model.predict(X[:5])
print(f"Sample predictions: {predicted}")
```

Pickle jest łatwy w użyciu, ale działa tylko w środowisku Python i nie jest odpowiedni do interoperacyjności między różnymi platformami.

---

### **ONNX**

ONNX (Open Neural Network Exchange) to otwarty format umożliwiający eksportowanie modeli do użycia na różnych platformach i w różnych językach. Jest szczególnie przydatny, gdy modele muszą być wdrażane w aplikacjach nie korzystających bezpośrednio z Pythona.

#### Przykład: Eksportowanie modelu do ONNX (przykładowy kod)

ONNX wymaga instalacji dodatkowych bibliotek, takich jak `onnxmltools` i `skl2onnx`.

```python
# Instalacja dodatkowych pakietów (uruchamiana w terminalu, nie w kodzie):
# pip install skl2onnx onnx

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Eksport modelu do ONNX
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Zapis modelu ONNX do pliku
with open("iris_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model exported to ONNX format.")
```

ONNX zapewnia większą elastyczność w zastosowaniach międzyplatformowych, takich jak wdrażanie modeli w aplikacjach mobilnych lub w środowisku .NET.

### Podsumowanie

Integracja Scikit-learn z innymi narzędziami, takimi jak Pandas, Matplotlib, Seaborn, Pickle i ONNX, czyni tę bibliotekę jeszcze bardziej potężną. **Pandas** umożliwia kompleksową manipulację danymi, **Matplotlib** i **Seaborn** pozwalają na zaawansowaną wizualizację wyników, a **Pickle** i **ONNX** oferują różne możliwości przechowywania i wdrażania modeli. Te narzędzia razem stanowią fundament nowoczesnego workflow analizy danych i budowy modeli uczenia maszynowego.

---
