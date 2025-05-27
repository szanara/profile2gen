import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch import nn

class DataIQ_Torch:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self._predicted_values = None
        self._true_values = y
        self._grads = None

    def gradient(self, net, device):
        net.to(device)
        net.train()
        data = torch.tensor(self.X, device=device, dtype=torch.float32)
        targets = torch.tensor(self.y, device=device, dtype=torch.float32)
        loss_fn = torch.nn.MSELoss()

        activations = {}
        handles = []

        def save_activations_hook(layer):
            def hook(module, input, output):
                activations[layer] = input[0].detach()
            return hook

        def per_example_norms_hook(layer):
            def hook(module, grad_input, grad_output):
                A = activations[layer]
                B = grad_output[0]
                norms[0] += (A * A).sum(dim=1) * (B * B).sum(dim=1)
            return hook

        # Register forward hooks to save activations
        for layer in net.children():
            handles.append(layer.register_forward_hook(save_activations_hook(layer)))

        # Forward pass
        output = net(data)
        loss = loss_fn(output, targets)

        # Prepare norms array
        norms = [torch.zeros(data.shape[0], device=device)]

        # Register backward hooks to compute norms
        for layer in net.children():
            handles.append(layer.register_backward_hook(per_example_norms_hook(layer)))

        # Backward pass
        loss.backward()

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Get per-example gradient norms
        grads_train = norms[0].cpu().numpy()

        if self._grads is None:
            self._grads = np.expand_dims(grads_train, axis=-1)
        else:
            stack = [self._grads, np.expand_dims(grads_train, axis=-1)]
            self._grads = np.hstack(stack)

    def on_epoch_end(self, net, device="cuda", **kwargs):
        self.gradient(net, device)
        predicted_values = []

        net.eval()
        with torch.no_grad():
            for i in range(len(self.X)):
                x = torch.tensor(self.X[i, :], device=device, dtype=torch.float32)
                pred = net(x.unsqueeze(0)).item()
                predicted_values.append(pred)

        predicted_values = np.array(predicted_values)

        if self._predicted_values is None:
            self._predicted_values = np.expand_dims(predicted_values, axis=-1)
        else:
            stack = [self._predicted_values, np.expand_dims(predicted_values, axis=-1)]
            self._predicted_values = np.hstack(stack)

    @property
    def get_grads(self):
        return self._grads

    @property
    def predicted_values(self):
        return self._predicted_values

    @property
    def true_values(self):
        return self._true_values

    @property
    def confidence(self):
        return np.mean(self._predicted_values, axis=-1)

    @property
    def aleatoric(self):
        return np.var(self._predicted_values, axis=-1)

    @property
    def variability(self):
        return np.std(self._predicted_values, axis=-1)

    @property
    def correctness(self):
        return np.sqrt(np.mean((self._predicted_values - self._true_values[:, None]) ** 2, axis=-1))

    @property
    def entropy(self):
        ss_res = np.sum((self._predicted_values - self._true_values[:, None]) ** 2, axis=-1)
        ss_tot = np.sum((self._true_values[:, None] - np.mean(self._true_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    @property
    def mi(self):
        true_vals = self._true_values[:, None]
        pred_vals = self._predicted_values
        correlation_matrix = np.corrcoef(pred_vals.flatten(), true_vals.flatten())
        correlation_coeff = correlation_matrix[0, 1]
        return correlation_coeff

def fit_dataiq(X, y, device, epochs, net):
    net.to(device)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).cuda()
    y_tensor = torch.tensor(y, dtype=torch.float32).cuda()
    train_data = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True)
    dataiq = DataIQ_Torch(X_scaled, y)
    EPOCHS = epochs

    optimizer = torch.optim.Adam(net.parameters())

    for e in tqdm(range(1, EPOCHS + 1)):
        net.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = net(X_batch)
            loss = nn.MSELoss()(output, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
        dataiq.on_epoch_end(net, device=device)

    aleatoric_uncertainty = dataiq.aleatoric
    confidence = dataiq.confidence
    return aleatoric_uncertainty, confidence

class DataIQ_SKLearn:
    def __init__(self, X, y):
        """
        The function takes in the training data and the labels, and stores them in the class variables X
        and y. It also stores the boolean value of sparse_labels in the class variable _sparse_labels

        Args:
          X: the input data
          y: the true labels
        """
        self.X = X
        self.y = np.asarray(y)
        
        # placeholder
        self._predicted_values = None
        self._true_values = self.y

    def on_epoch_end(self, clf, device="cuda", iteration=1, **kwargs):
        """
        The function computes the predicted and true values over all samples in the
        dataset for each epoch.

        Args:
          clf: the regressor object
          device: The device to use for the computation. Defaults to cpu
          iteration: The current iteration of the training loop. Defaults to 1
        """
        # Compute the predicted values over all samples in the dataset
        predicted_values = []

        x = self.X

        # Predict using the regressor
        predictions = clf.predict(x)
        
        predicted_values = np.array(predictions)

        # Append the new predicted values
        if self._predicted_values is None:  # On first epoch of training
            self._predicted_values = np.expand_dims(predicted_values, axis=-1)
        else:
            stack = [
                self._predicted_values,
                np.expand_dims(predicted_values, axis=-1),
            ]
            self._predicted_values = np.hstack(stack)

    @property
    def predicted_values(self) -> np.ndarray:
        """
        Returns:
            Predicted values across epochs: np.array(n_samples, n_epochs)
        """
        return self._predicted_values

    @property
    def true_values(self) -> np.ndarray:
        """
        Returns:
            True values (actual labels): np.array(n_samples)
        """
        return self._true_values

    @property
    def confidence(self) -> np.ndarray:
        """
        Returns:
            Average predictive confidence (mean predictions) across epochs: np.array(n_samples)
        """
        return np.mean(self._predicted_values, axis=-1)

    @property
    def aleatoric(self):
        """
        Returns:
            Aleatoric uncertainty (variance of predictions) across epochs: np.array(n_samples)
        """
        return np.var(self._predicted_values, axis=-1)

    @property
    def variability(self) -> np.ndarray:
        """
        Returns:
            Epistemic variability (standard deviation of predictions) across epochs: np.array(n_samples)
        """
        return np.std(self._predicted_values, axis=-1)

    @property
    def correctness(self) -> np.ndarray:
        """
        Returns:
            Root mean squared error between predictions and true values across epochs: np.array(n_samples)
        """
        return np.sqrt(np.mean((self._predicted_values - self._true_values[:, None]) ** 2, axis=-1))

    @property
    def entropy(self):
        """
        Returns:
            Coefficient of determination (R^2 score) across epochs: np.array(n_samples)
        """
        ss_res = np.sum((self._predicted_values - self._true_values[:, None]) ** 2, axis=-1)
        ss_tot = np.sum((self._true_values[:, None] - np.mean(self._true_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    @property
    def mi(self):
        """
        Returns:
            Mutual information (correlation) between predictions and true values across epochs: np.array(n_samples)
        """
        true_vals = self._true_values[:, None]
        pred_vals = self._predicted_values
        correlation_matrix = np.corrcoef(pred_vals.flatten(), true_vals.flatten())
        correlation_coeff = correlation_matrix[0, 1]
        return correlation_coeff

def fit_dataiq_sk(X, y, device, epochs, clf):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    dataiq = DataIQ_SKLearn(X_scaled, y)

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    for e in range(1, epochs + 1):
        # Treinar o modelo com uma iteração
        clf.fit(X_train, y_train)
        dataiq.on_epoch_end(clf, device=device, iteration=e)

    aleatoric_uncertainty = dataiq.aleatoric
    confidence = dataiq.confidence
    return aleatoric_uncertainty, confidence



def filter_with_dataiq(X_train, y_train,aleatoric_uncertainty, confidence, threshold=0.5):
    """
    It takes in the training data and labels, and the number of trees in the XGBoost model, and returns
    the indices of the easy, ambiguous, and hard training examples, as well as the aleatoric uncertainty
    of each training example

    Args:
      X_train: the training data
      y_train: the labels for the training data
      nest: number of estimators

    Returns:
      the indices of the easy, ambiguous, and hard training examples.
    """
    percentile_thresh = threshold
    conf_thresh_low = threshold - threshold//2
    conf_thresh_high = threshold + threshold//2
    confidence_train = confidence
    aleatoric_train = aleatoric_uncertainty
    # Get the 3 subgroups
    hard_train = np.where(
        (confidence_train <= conf_thresh_low)
        & (aleatoric_train <= np.percentile(aleatoric_train, percentile_thresh)),
    )[0]
    easy_train = np.where(
        (confidence_train >= conf_thresh_high)
        & (aleatoric_train <= np.percentile(aleatoric_train, percentile_thresh)),
    )[0]

    hard_easy = np.concatenate((hard_train, easy_train))
    ambig_train = []
    for id in range(len(confidence_train)):
        if id not in hard_easy:
            ambig_train.append(id)
    ambig_train = np.array(ambig_train)

    filtered_ids = np.concatenate((easy_train, ambig_train))  # filtered ids
    return easy_train, ambig_train, hard_train, aleatoric_train, filtered_ids

