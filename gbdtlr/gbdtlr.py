import numpy as np
import warnings
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


class GBDTFeatureGenerator(BaseEstimator, TransformerMixin):

	def __init__(self, 
				n_estimators=10,
				num_leaves=16,
				model_type='lightgbm',
				categorical_cols=None,
				include_original_feature=True,
				random_state=None):
		if model_type not in ('lightgbm', 'gbdt', 'forest'):
			raise ValueError('Only {} is supported for now'.format(['lightgbm', 'gbdt', 'forest']))

		self.n_estimators = n_estimators
		self.num_leaves = num_leaves
		self.model_type = model_type
		self.categorical_cols = categorical_cols
		self.include_original_feature = include_original_feature
		self.random_state = random_state

	def _build_lgbm(self, X, y):
		from lightgbm import LGBMClassifier

		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			model = LGBMClassifier(n_estimators=self.n_estimators, 
									num_leaves=self.num_leaves,
									random_state=self.random_state)
			model.fit(X, y, categorical_feature=self.categorical_cols or [])
		return model

	def _build_gbdt(self, X, y):
		from sklearn.ensemble import GradientBoostingClassifier
		model = GradientBoostingClassifier(n_estimators=self.n_estimators,
											max_leaf_nodes=self.num_leaves,
											random_state=self.random_state)
		model.fit(X, y)
		return model

	def _build_forest(self, X, y):
		from sklearn.ensemble import RandomForestClassifier
		model = RandomForestClassifier(n_estimators=self.n_estimators,
										max_leaf_nodes=self.num_leaves,
										random_state=self.random_state)
		model.fit(X, y)
		return model                                                                                                                                                                                                         

	def fit(self, X, y):
		if self.model_type == 'lightgbm':
			self.model = self._build_lgbm(X, y)
		elif self.model_type == 'gbdt':
			self.model = self._build_gbdt(X, y)
		elif self.model_type == 'forest':
			self.model = self._build_forest(X, y)
		return self

	def _transform_lgbm(self, X):
		lgb = self.model
		
		leaf_index = lgb.predict(X, pred_leaf=True)
		start_index = np.concatenate([np.arange(self.n_estimators)[np.newaxis, ...]] * X.shape[0], axis=0) * self.num_leaves
		leaf_index += start_index

		encoded = np.zeros((X.shape[0], self.n_estimators * self.num_leaves))
		for row, loc in enumerate(leaf_index):
		    encoded[row, loc] += 1
		return encoded

	def _transform_scikit(self, X):
		# TODO: the leaf index of scikit tree models is not continuous
		# therefore introduces a bunch of all-zero columns making the matrix sparse
		model = self.model

		num_nodes_per_tree = np.array([t.tree_.node_count for t in model.estimators_])
		num_nodes = sum(num_nodes_per_tree)

		leaf_index = model.apply(X)
		start_index = np.insert(np.cumsum(num_nodes_per_tree), 0, 0)[:-1]
		start_index = np.concatenate([start_index[np.newaxis, ...]] * X.shape[0], axis=0)
		leaf_index += start_index

		encoded = np.zeros((X.shape[0], num_nodes))
		for row, loc in enumerate(leaf_index):
		    encoded[row, loc] += 1
		return encoded

	def transform(self, X):
		if not hasattr(self, 'model'):
			raise NotFittedError('{} is not fitted.'.format(self.__class__))

		if self.model_type == 'lightgbm':
			encoded = self._transform_lgbm(X)
		else:
			encoded = self._transform_scikit(X)

		if self.include_original_feature:
			encoded = np.concatenate([X, encoded], axis=1)

		return encoded			


def GBDTLR(
			# arguments for feature generator
			n_estimators=10,
			num_leaves=16,
			model_type='lightgbm',
			categorical_cols=None,
			include_original_feature=True,
			# arguments for logistic regression
			penalty='l2',
			tol=0.0001,
			C=1.0,
			fit_intercept=True,
			intercept_scaling=1,
			class_weight=None,
			max_iter=100,
			solver='liblinear',
			random_state=None):
		feature_generator = GBDTFeatureGenerator(n_estimators=n_estimators,
												num_leaves=num_leaves,
												model_type=model_type,
												categorical_cols=categorical_cols,
												include_original_feature=include_original_feature,
												random_state=random_state)
		lr = LogisticRegression(penalty=penalty,
								tol=tol,
								C=C,
								fit_intercept=fit_intercept,
								intercept_scaling=intercept_scaling,
								class_weight=class_weight,
								max_iter=max_iter,
								solver=solver,
								random_state=random_state)

		return make_pipeline(feature_generator, lr)