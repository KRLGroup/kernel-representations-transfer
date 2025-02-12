import numpy as np
from sklearn.decomposition import KernelPCA
from FiniteStateMachine import MooreMachine
import matplotlib.pyplot as plt

class KernelPCA_DFA:
    def __init__(self, dfas, max_step, threshold):
        """
        Inizializza la classe con un insieme di oggetti DFA.
        :param dfas: Lista di oggetti DFA
        """
        self.dfas = dfas
        self.n = len(dfas)
        self.threshold = threshold
        self.max_step_similarity = max_step

        self.fit_transform_optimal_num_components()

    def compute_kernel_matrix(self, dfas1, dfas2):
        """
        Calcola la matrice del kernel tra due liste di DFA.
        :param dfas1: Prima lista di DFA
        :param dfas2: Seconda lista di DFA
        :return: Matrice del kernel
        """
        K = np.zeros((len(dfas1), len(dfas2)))
        for i in range(len(dfas1)):
            for j in range(len(dfas2)):
                K[i, j] = dfas1[i].similarity(dfas2[j], max_step=self.max_step_similarity)
        return K

    def fit_transform_optimal_num_components(self):
        """
        Esegue Kernel PCA sulla matrice del kernel calcolata.
        :param n_components: Numero di componenti principali da mantenere
        :return: Trasformazione dei DFA nello spazio PCA
        """
        self.K_fit = self.compute_kernel_matrix(self.dfas, self.dfas)

        self.n_components = self.choose_n_components()

        self.kpca = KernelPCA(n_components=self.n_components, kernel='precomputed')
        self.kpca.fit_transform(self.K_fit)

        return

    def transform(self, new_dfas):
        """
        Proietta nuovi DFA nello spazio Kernel PCA già calcolato.
        :param new_dfas: Lista di nuovi oggetti DFA
        :return: Proiezione dei nuovi DFA nello spazio PCA
        """
        K_new = self.compute_kernel_matrix(new_dfas, self.dfas)  # Kernel tra nuovi DFA e quelli di training
        return self.kpca.transform(K_new)

    def choose_n_components(self):
        """
        Determina il numero ottimale di componenti principali da mantenere in base alla varianza spiegata cumulativa.
        :param threshold: Soglia di varianza spiegata cumulativa (default 95%)
        :return: Numero ottimale di componenti principali
        """
        threshold = self.threshold
        if self.K_fit is None:
            raise ValueError("Devi prima calcolare la matrice del kernel con 'fit_transform'.")

        # Decomposizione della matrice del kernel
        eigenvalues, _ = np.linalg.eigh(self.K_fit)
        eigenvalues = eigenvalues[::-1]  # Ordinati in ordine decrescente

        # Calcolo della varianza spiegata cumulativa
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Trova il numero minimo di componenti per superare la soglia
        n_components = np.searchsorted(cumulative_variance, threshold) + 1

        # Plot della varianza spiegata
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold * 100}% Varianza Spiegata')
        plt.xlabel("Numero di Componenti")
        plt.ylabel("Varianza Spiegata Cumulativa")
        plt.title("Selezione del Numero di Componenti in Kernel PCA")
        plt.legend()
        plt.savefig("kernel_pca_variance.png")
        plt.close()

        return n_components
def plot_transformed_data(transformed_train_data, transformed_test_data=None, img_path = "kernel_pca.png"):
        """
        Plotta i dati trasformati nello spazio Kernel PCA.
        :param transformed_train_data: Dati trasformati di training
        :param transformed_test_data: (Opzionale) Dati trasformati di test
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(transformed_train_data[:, 0], transformed_train_data[:, 1], color='blue', label='Train')

        if transformed_test_data is not None:
            plt.scatter(transformed_test_data[:, 0], transformed_test_data[:, 1], color='red', label='Test')

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Kernel PCA Projection')
        plt.legend()
        plt.savefig(img_path)
        plt.close()

if __name__ == '__main__':

    # Esempio di utilizzo:
    alphabet = ['a', 'b', 'c', 'd']
    max_step_similarity = 4
    threshold = 0.95

    training_formulas = ['F a', 'F b', 'F a & F c', 'F d & F b', 'F a & F b & F c', 'F d & F c & F a']
    test_formulas = ['F c', 'F d', 'F d & F b', 'F b & F a']

    dfa_train = []
    dfa_test = []
    for formula in training_formulas:
        dfa_train.append(MooreMachine(formula, None, None, reward="acceptance", dictionary_symbols=alphabet))

    for formula in test_formulas:
        dfa_test.append(MooreMachine(formula, None,None, reward="acceptance", dictionary_symbols=alphabet))

    kpca_dfa = KernelPCA_DFA(dfa_train, max_step_similarity, threshold=threshold)

    transformed_train_data = kpca_dfa.transform(dfa_train)
    transformed_test_data = kpca_dfa.transform(dfa_test)

    print(transformed_train_data)
    print(transformed_test_data)

    plot_transformed_data(transformed_train_data, transformed_test_data, "kernel_pca.png")