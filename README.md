# Przewidywanie dochodu (Adult Census Income) - Projekt Sieci Neuronowe

### Autorzy
* **Mateusz Hypta** (280116)
* **Mateusz Kwapisz** (280107)

### Opis projektu
Projekt ma na celu stworzenie modelu sieci neuronowej (MLP), który na podstawie danych demograficznych przewiduje, czy roczny dochód danej osoby przekracza 50,000 USD. Problem klasyfikacji binarnej rozwiązano przy użyciu biblioteki TensorFlow/Keras, stosując techniki takie jak Oversampling oraz Grid Search do optymalizacji parametrów.

### Struktura repozytorium
* `src/` - Kody źródłowe (trening modelu, ewaluacja, generowanie wykresów).
* `modele/` - Zbiór uzyskanych historycznych modeli.
* `wykresy/` - Wygenerowane wykresy (Krzywa uczenia, ROC, Macierz pomyłek).

### Wyniki
Ostateczny model osiągnął następujące wyniki na zbiorze testowym:
* **Accuracy:** ~84.4%
* **AUC:** ~0.92
* **Recall (dla klasy >50K):** ~85.2%

### Jak uruchomić projekt
1. Zainstaluj wymagane biblioteki:
   pip install -r requirements.txt