# **4. Przykładowe zastosowania Scikit-learn**

Scikit-learn jest jedną z najbardziej wszechstronnych bibliotek Python, oferującą szeroki wachlarz algorytmów i narzędzi do rozwiązywania problemów uczenia maszynowego. W tym rozdziale szczegółowo omówimy kluczowe zastosowania Scikit-learn w obszarach takich jak klasyfikacja, regresja, klastrowanie, redukcja wymiarowości oraz wykrywanie anomalii. Każdy podrozdział zawiera praktyczne przykłady, ilustrujące możliwości i elastyczność tej biblioteki.


## **4.1. Klasyfikacja**

Klasyfikacja to technika polegająca na przypisywaniu danych wejściowych do jednej z kilku z góry określonych kategorii. Jest ona szeroko stosowana w zadaniach takich jak rozpoznawanie obrazów, analiza tekstu czy diagnostyka medyczna.

### **Przykład: Rozpoznawanie irysów**

Jednym z najpopularniejszych zestawów danych w uczeniu maszynowym jest zbiór **Iris**, zawierający dane o trzech gatunkach irysów: *setosa*, *versicolor* i *virginica*. Każdy rekord opisuje cztery cechy: długość i szerokość działki kielicha oraz długość i szerokość płatka.

#### Implementacja:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Załadowanie danych
iris = load_iris()
X, y = iris.data, iris.target

# Podział danych na treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trenowanie drzewa decyzyjnego
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Przewidywanie i ocena
y_pred = model.predict(X_test)
print(f"Dokładność klasyfikacji: {accuracy_score(y_test, y_pred):.2f}")
```

#### **Popularne algorytmy:**
1. **Drzewa decyzyjne:** Intuicyjne, szybkie i łatwe do interpretacji.
2. **Support Vector Machines (SVM):** Idealne do analizy danych wielowymiarowych, dobrze radzące sobie z małymi zbiorami danych.
3. **k-Nearest Neighbors (kNN):** Metoda oparta na podobieństwie próbek, skuteczna przy niewielkich zbiorach danych.

Drzewa decyzyjne mogą być używane do prostych klasyfikacji, ale w bardziej złożonych przypadkach warto sięgnąć po algorytmy takie jak SVM, które wykorzystują hiperplany w przestrzeniach wielowymiarowych, aby rozdzielić klasy.

---

## **4.2. Regresja**

Regresja to technika używana do przewidywania wartości liczbowych. Często stosowana w analizie cen, przewidywaniu wartości giełdowych czy analizie trendów.

### **Przykład: Przewidywanie cen nieruchomości**

Załóżmy, że chcemy przewidzieć cenę nieruchomości na podstawie jej powierzchni, liczby pokoi i lokalizacji. Możemy użyć generowanego zestawu danych lub rzeczywistych danych nieruchomościowych.

#### Implementacja:

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generowanie danych
X, y = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trenowanie regresji liniowej
model = LinearRegression()
model.fit(X_train, y_train)

# Przewidywanie i ocena
y_pred = model.predict(X_test)
print(f"Średni błąd kwadratowy: {mean_squared_error(y_test, y_pred):.2f}")
```

#### **Popularne algorytmy:**
1. **Regresja liniowa:** Najprostsza metoda regresji, zakładająca liniową zależność między zmiennymi.
2. **Lasso Regression:** Regresja liniowa z regularizacją L1, skuteczna w eliminowaniu nieistotnych zmiennych.
3. **Ridge Regression:** Regresja liniowa z regularizacją L2, zapobiegająca przeuczeniu modelu.

Regresja liniowa dobrze sprawdza się w prostych zadaniach, ale w przypadku bardziej złożonych zależności warto rozważyć algorytmy regularizowane.

---

## **4.3. Klastrowanie**

Klastrowanie to metoda grupowania danych w klastry bez potrzeby wcześniejszego oznaczania ich etykietami. Jest szeroko stosowana w analizie klientów, eksploracji danych i segmentacji obrazów.

### **Przykład: Segmentacja klientów**

Załóżmy, że chcemy podzielić klientów na grupy na podstawie ich wieku i miesięcznych wydatków. Klastrowanie pomoże zidentyfikować grupy klientów o podobnych cechach.

#### Implementacja:

```python
from sklearn.cluster import KMeans
import numpy as np

# Przykładowe dane (wiek, wydatki)
X = np.array([[25, 400], [40, 500], [35, 700], [50, 200], [45, 300], [30, 800]])

# Klastrowanie K-means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Wyświetlenie etykiet klastrów
print(f"Etykiety klastrów: {kmeans.labels_}")
```

#### **Popularne algorytmy:**
1. **K-means:** Prosty i szybki algorytm dzielący dane na zdefiniowaną liczbę klastrów.
2. **Hierarchiczne klastrowanie:** Tworzy hierarchiczne drzewa (dendrogramy) przedstawiające strukturę klastrów.

Hierarchiczne klastrowanie jest bardziej intuicyjne w analizie danych o złożonej strukturze, natomiast K-means jest efektywniejsze przy dużych zbiorach danych.

---

## **4.4. Redukcja wymiarowości**

Redukcja wymiarowości pozwala zmniejszyć liczbę cech w zbiorze danych przy zachowaniu istotnych informacji. Technika ta jest szczególnie przydatna w analizie danych wielowymiarowych, takich jak dane genetyczne, dane finansowe czy obrazy.

### **Przykład: Analiza danych obrazowych**

W przypadku analizy obrazów, gdzie każdy obraz jest reprezentowany przez tysiące pikseli, redukcja wymiarowości może uprościć dane do bardziej przystępnej formy, zachowując kluczowe informacje.

#### Implementacja z PCA:

```python
from sklearn.decomposition import PCA
import numpy as np

# Przykładowe dane (5 próbek, 10 cech)
X = np.random.rand(5, 10)

# Redukcja wymiarowości do 2D
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"Zredukowane dane: {X_reduced}")
```

#### Implementacja z t-SNE:

```python
from sklearn.manifold import TSNE

# Redukcja wymiarowości za pomocą t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X)

print(f"Zredukowane dane: {X_embedded}")
```

#### **Popularne techniki:**
1. **Principal Component Analysis (PCA):** Analiza głównych składowych identyfikuje cechy o największej wariancji.
2. **t-SNE:** Technika wizualizacji danych w niskowymiarowej przestrzeni, np. 2D lub 3D.

PCA jest szybsze i bardziej efektywne przy dużych zbiorach danych, podczas gdy t-SNE lepiej oddaje lokalne zależności między punktami.

---

## **4.5. Wykrywanie anomalii**

Wykrywanie anomalii to technika identyfikacji nietypowych danych, takich jak oszustwa finansowe, błędy w systemach czy niespójności w logach.

### **Przykład: Identyfikacja oszustw finansowych**

Dane transakcji mogą zawierać anomalie, które reprezentują oszustwa. Algorytmy wykrywania anomalii mogą pomóc w ich identyfikacji.

#### Implementacja z Isolation Forest:

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# Dane (większość próbek jest normalna, kilka to anomalie)
X = np.array([[

10], [12], [10], [11], [100], [12], [9]])

# Wykrywanie anomalii
isolation_forest = IsolationForest(random_state=42)
isolation_forest.fit(X)

# Ocena
anomalies = isolation_forest.predict(X)
print(f"Anomalie: {anomalies}")
```

#### Implementacja z One-class SVM:

```python
from sklearn.svm import OneClassSVM

# Wykrywanie anomalii za pomocą One-class SVM
svm = OneClassSVM()
svm.fit(X)
outliers = svm.predict(X)
print(f"Anomalie: {outliers}")
```

#### **Popularne algorytmy:**
1. **Isolation Forest:** Skuteczny algorytm do wykrywania anomalii w dużych zbiorach danych.
2. **One-class SVM:** Algorytm uczony na danych „normalnych” do identyfikacji odchyleń.


### Podsumowanie

Scikit-learn oferuje wszechstronne narzędzia do rozwiązywania złożonych problemów uczenia maszynowego w różnych dziedzinach, od klasyfikacji i regresji, przez klastrowanie, aż po redukcję wymiarowości i wykrywanie anomalii. Dzięki spójności API i różnorodności algorytmów, biblioteka ta pozwala na efektywne eksperymentowanie, testowanie i wdrażanie modeli. Każdy z opisanych obszarów znajduje szerokie zastosowanie w rzeczywistych scenariuszach, czyniąc Scikit-learn kluczowym narzędziem w arsenale każdego specjalisty ds. danych.

---
