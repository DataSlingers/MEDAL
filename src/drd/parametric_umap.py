from umap.parametric_umap import ParametricUMAP

class ParametricUMAPTorch:
    """
    A wrapper for the ParametricUMAP class from umap.parametric_umap.
    This class is used to create a parametric UMAP model for dimensionality reduction.
    """
    def __init__(self, autoencoder_loss=True, **kwargs):
        """
        Initialize the ParametricUMAP model with the given parameters.
        
        Parameters:
        - **kwargs: Additional keyword arguments to pass to the ParametricUMAP constructor.
        """
        self.p_umap = ParametricUMAP(
            parametric_reconstruction=True,
            autoencoder_loss = autoencoder_loss, 
            **kwargs)
    
    def fit(self, X, y=None):  
        """
        Fit the ParametricUMAP model to the data.
        
        Parameters:
        - X: Input data.
        - y: Optional labels for the data.
        """
        super().fit(X, y)

    def transform(self, X):
        return super().transform(X)
    
    def inverse_transform(self, Z):
        """
        Inverse transform the latent space back to the original space.
        
        Parameters:
        - Z: Latent space representation.
        
        Returns:
        - X_recon: Reconstructed data from the latent space.
        """
        return super().inverse_transform(Z)
    
