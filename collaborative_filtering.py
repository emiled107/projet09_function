import pandas as pd
import numpy as np
import logging
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn  # Pour l'intégration avec MLflow
import joblib  # Pour la sauvegarde du modèle

class CollaborativeFiltering:
    def __init__(self, articles_metadata_path, clicks_path, n_components=100):
        self.articles_metadata_path = articles_metadata_path
        self.clicks_path = clicks_path
        self.n_components = n_components
        self.user_article_matrix = None
        self.svd_model = None
        self.clicks_df = None
        self.articles_metadata_df = None

    def load_data(self):
        # Chargement des métadonnées des articles
        if self.articles_metadata_path is not None:
            articles_metadata_df = pd.read_csv(self.articles_metadata_path)
        else:
            articles_metadata_df = None

        # Vérifier si clicks_path est un répertoire ou un fichier
        if os.path.isdir(self.clicks_path):
            # Lire tous les fichiers CSV dans le répertoire
            clicks_df = pd.concat(
                [pd.read_csv(os.path.join(self.clicks_path, f)) for f in os.listdir(self.clicks_path) if f.endswith('.csv')],
                ignore_index=True
            )
        elif os.path.isfile(self.clicks_path):
            # Charger directement le fichier si c'est un fichier unique
            clicks_df = pd.read_csv(self.clicks_path)
        else:
            raise FileNotFoundError(f"Aucun fichier ou répertoire valide trouvé à {self.clicks_path}")

        # Nettoyage des données de clics
        clicks_df.drop_duplicates(inplace=True)
        clicks_df.dropna(inplace=True)

        return clicks_df, articles_metadata_df

    def clean_and_prepare_data(self, clicks_df):
        clicks_df.drop_duplicates(inplace=True)
        clicks_df.dropna(inplace=True)
        return clicks_df

    def build_interaction_matrix(self, clicks_df):
        # Encodage des IDs d'utilisateurs et d'articles
        user_encoder = LabelEncoder()
        article_encoder = LabelEncoder()
        
        clicks_df['user_id_encoded'] = user_encoder.fit_transform(clicks_df['user_id'])
        clicks_df['article_id_encoded'] = article_encoder.fit_transform(clicks_df['click_article_id'])
        
        # Création de la matrice d'interaction
        interaction_matrix = pd.pivot_table(clicks_df, index='user_id_encoded', columns='article_id_encoded', aggfunc='size', fill_value=0)
        
        return interaction_matrix

    def train_model(self, interaction_matrix):
        self.user_article_matrix = interaction_matrix
        num_features = interaction_matrix.shape[1]
        n_components = min(self.n_components, num_features - 1)
        self.svd_model = TruncatedSVD(n_components=n_components)
        self.svd_model.fit(interaction_matrix)
        logging.info("Modèle entraîné avec succès")

    def evaluate_model(self, interaction_matrix):
        # Vérifiez que le modèle est chargé
        if self.svd_model is None:
            logging.error("Le modèle n'est pas chargé.")
            raise ValueError("Le modèle n'est pas chargé.")

        # Créez une matrice d'interactions utilisateur-article pour évaluer le modèle
        num_users, num_items = interaction_matrix.shape
        all_predictions = np.zeros((num_users, num_items))

        # Prédictions pour chaque utilisateur
        for user_index in range(num_users):
            user_vector = interaction_matrix.iloc[user_index, :].values.reshape(1, -1)
            predicted_vector = self.predict(user_index, user_vector)  # Prédiction pour un utilisateur
            all_predictions[user_index, :] = predicted_vector

        # Récupérer les valeurs réelles
        true_values = interaction_matrix.to_numpy().flatten()
        predicted_values = all_predictions.flatten()

        # Calculer les métriques
        rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
        mae = mean_absolute_error(true_values, predicted_values)
        r2 = r2_score(true_values, predicted_values)
        correlation, _ = pearsonr(true_values, predicted_values)

        return rmse, mae, r2, correlation


    def save_model(self, filename='svd_model.joblib'):
        if self.svd_model:
            joblib.dump(self.svd_model, filename)
            logging.info(f'Modèle sauvegardé sous {filename}')
        else:
            logging.error("Aucun modèle à sauvegarder.")

    def load_model(self, filename='svd_model.joblib'):
        self.svd_model = joblib.load(filename)
        logging.info(f'Modèle chargé depuis {filename}')

    def load_model_df(self, svd_model):
        self.svd_model = svd_model
        logging.info(f'Modèle svd chargé')

    def predict(self, user_index, user_interaction_matrix):
        """ Fait des prédictions basées sur la matrice d'interaction de l'utilisateur. """
        if self.svd_model:
            # Si user_interaction_matrix est un DataFrame, convertir en ndarray
            if isinstance(user_interaction_matrix, pd.DataFrame):
                user_interaction_matrix = user_interaction_matrix.values

            predictions = self.svd_model.transform(user_interaction_matrix)
            return self.svd_model.inverse_transform(predictions).flatten()
        else:
            logging.error("Aucun modèle chargé pour faire des prédictions.")
            return np.zeros(user_interaction_matrix.shape[1])

    def update_data(self):
        """ Mise à jour de la matrice d'interaction avec de nouvelles interactions utilisateur-article. """
        clicks_df, _ = self.load_data()
        clicks_df = self.clean_and_prepare_data(clicks_df)
        new_interaction_matrix = self.build_interaction_matrix(clicks_df)
        if self.user_article_matrix is None:
            self.user_article_matrix = new_interaction_matrix
        else:
            self.user_article_matrix += new_interaction_matrix
        logging.info("Données mises à jour avec de nouvelles interactions.")

    def load_user_article_matrix_df(self, matrix_df):
        self.user_article_matrix = matrix_df
        
        # Si nécessaire, renommer la première colonne en 'user_id' (au cas où le nom serait différent)
        self.user_article_matrix.rename(columns={self.user_article_matrix.columns[0]: 'user_id'}, inplace=True)
        
        logging.info(f"Matrice d'interaction chargée")
        logging.info(f"Colonnes de la matrice : {self.user_article_matrix.columns}")
        logging.info(f"Shape de la matrice : {self.user_article_matrix.shape}")
        logging.info(f"Premières lignes de la matrice :\n{self.user_article_matrix.head()}")

    def load_user_article_matrix(self, matrix_path):
        # Charger le CSV en spécifiant que la première colonne (user_id) ne doit pas être utilisée comme index
        self.user_article_matrix = pd.read_csv(matrix_path, index_col=None)
        
        # Si nécessaire, renommer la première colonne en 'user_id' (au cas où le nom serait différent)
        self.user_article_matrix.rename(columns={self.user_article_matrix.columns[0]: 'user_id'}, inplace=True)
        
        logging.info(f"Matrice d'interaction chargée depuis {matrix_path}")
        logging.info(f"Colonnes de la matrice : {self.user_article_matrix.columns}")
        logging.info(f"Shape de la matrice : {self.user_article_matrix.shape}")
        logging.info(f"Premières lignes de la matrice :\n{self.user_article_matrix.head()}")


    def save_user_article_matrix(self, matrix_path):
        """ Sauvegarde la matrice d'interaction utilisateur-article dans un fichier. """
        if self.user_article_matrix is not None:
            self.user_article_matrix.to_csv(matrix_path)
            logging.info(f"Matrice d'interaction sauvegardée sous {matrix_path}")
        else:
            logging.error("Aucune matrice d'interaction à sauvegarder.")

    def retrain_model(self):
        """ Réentraînement du modèle SVD avec la matrice d'interaction mise à jour. """
        if self.user_article_matrix is not None and not self.user_article_matrix.empty:
            num_features = self.user_article_matrix.shape[1]
            n_components = min(self.n_components, num_features - 1)
            self.svd_model = TruncatedSVD(n_components=n_components)
            self.svd_model.fit(self.user_article_matrix)
            logging.info("Modèle réentraîné avec succès.")
        else:
            logging.error("Aucune matrice d'interaction pour réentraîner le modèle.")

    def recommend_articles(self, user_id, top_n=5):
        if 'user_id' not in self.user_article_matrix.columns:
            raise ValueError("La colonne 'user_id' n'existe pas dans la matrice.")
        
        user_index = self.user_article_matrix.index[self.user_article_matrix['user_id'] == user_id].tolist()
        if not user_index:
            raise ValueError(f"user_id {user_id} non trouvé dans la matrice.")
        
        user_interaction_matrix = self.user_article_matrix.loc[user_index].drop('user_id', axis=1).values
        predictions = self.svd_model.transform(user_interaction_matrix)
        predictions = self.svd_model.inverse_transform(predictions)
        recommended_article_indices = np.argsort(-predictions.flatten())[:top_n]
        return self.user_article_matrix.columns[recommended_article_indices + 1].tolist()  # +1 pour sauter la colonne 'user_id'


    def run_pipeline(self):
        """ Pipeline complet d'entraînement du modèle avec MLflow. """
        mlflow.set_experiment('Collaborative_Filtering_Experiment')
        with mlflow.start_run():
            clicks_df = self.clean_and_prepare_data(clicks_df)
            interaction_matrix = self.build_interaction_matrix(clicks_df)
            self.train_model(interaction_matrix)
            rmse, mae, r2, correlation = self.evaluate_model(interaction_matrix)

            # Log metrics
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2_Score", r2)
            mlflow.log_metric("Correlation", correlation)

            self.save_model()  # Sauvegarder le modèle avec joblib et MLflow
            self.save_user_article_matrix('user_article_matrix.csv')
            mlflow.end_run()
