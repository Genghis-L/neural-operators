import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat


class NSData:
    """
    Unified Navier-Stokes data handler for FNO.
    
    Automatically converts data to FNO-compatible format [N, T, H, W].
    Provides convenient methods for training and prediction.
    
    Args:
        data: numpy array or torch tensor in [N, H, W, T] format (original)
        t_in: number of input time steps (default: 10)
        t_out: number of output time steps (default: 10)
        device: 'cuda' or 'cpu'
        
    Example:
        >>> ns_data = NSData(raw_data, device='cuda')
        >>> train_loader = ns_data.get_dataloader(batch_size=50)
        >>> pred = model(ns_data.get_input())
        >>> ns_data.plot_comparison(pred, sample_idx=0, time_idx=0)
    """
    
    def __init__(self, data, t_in=10, t_out=10, device='cpu'):
        self.t_in = t_in
        self.t_out = t_out
        self.device = device
        
        # Convert to tensor if numpy
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        
        # Original data is [N, H, W, T], convert to [N, T, H, W]
        if data.dim() == 4:
            self.data = data.permute(0, 3, 1, 2).to(device)
        else:
            self.data = data.to(device)
            
        self.n_samples = self.data.shape[0]
        self.n_times = self.data.shape[1]
        self.height = self.data.shape[2]
        self.width = self.data.shape[3]
        
    @property
    def shape(self):
        return self.data.shape
    
    def get_input(self, indices=None):
        """Get input data [N, t_in, H, W]"""
        if indices is None:
            return self.data[:, :self.t_in, :, :]
        return self.data[indices, :self.t_in, :, :]
    
    def get_output(self, indices=None):
        """Get output/target data [N, t_out, H, W]"""
        if indices is None:
            return self.data[:, self.t_in:self.t_in+self.t_out, :, :]
        return self.data[indices, self.t_in:self.t_in+self.t_out, :, :]
    
    def get_ground_truth(self, sample_idx=0, time_idx=0):
        """Get ground truth at specific sample and output time step"""
        return self.data[sample_idx, self.t_in + time_idx, :, :]
    
    def get_dataloader(self, batch_size=50, shuffle=True, indices=None):
        """Create DataLoader for training"""
        if indices is not None:
            input_data = self.get_input(indices)
            output_data = self.get_output(indices)
        else:
            input_data = self.get_input()
            output_data = self.get_output()
            
        dataset = TensorDataset(input_data, output_data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def split(self, train_size=1000):
        """Split data into train and test NSData objects"""
        train_data = NSData.__new__(NSData)
        train_data.data = self.data[:train_size]
        train_data.t_in = self.t_in
        train_data.t_out = self.t_out
        train_data.device = self.device
        train_data.n_samples = train_size
        train_data.n_times = self.n_times
        train_data.height = self.height
        train_data.width = self.width
        
        test_data = NSData.__new__(NSData)
        test_data.data = self.data[train_size:]
        test_data.t_in = self.t_in
        test_data.t_out = self.t_out
        test_data.device = self.device
        test_data.n_samples = self.n_samples - train_size
        test_data.n_times = self.n_times
        test_data.height = self.height
        test_data.width = self.width
        
        return train_data, test_data
    
    def __len__(self):
        return self.n_samples
    
    def __repr__(self):
        return f'NSData(samples={self.n_samples}, times={self.n_times}, size={self.height}x{self.width}, device={self.device})'


class MatlabFileReader:
    """
    A class for reading MATLAB files.

    Args:
        file_path (str): The path to the MATLAB file.
        device (str, optional): The device to store the data (default is 'cpu').
        to_numpy (bool, optional): Convert the data to NumPy array (default is True).
        to_tensor (bool, optional): Convert the data to PyTorch tensor (default is False).

    Methods:
        read_file(section='data'): Read the specified section of the MATLAB file.

    Attributes:
        file_path (str): The path to the MATLAB file.
        device (str): The device to store the data.
        to_numpy (bool): Convert the data to NumPy array.
        to_tensor (bool): Convert the data to PyTorch tensor.
    """
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.device = kwargs.get('device', 'cpu') # default is 'cpu'
        self.to_numpy = kwargs.get('to_numpy', True) # default is True
        self.to_tensor = kwargs.get('to_tensor', False) # default is False
        self._load_file_()
        
    def _load_file_(self):
        """
        Load the MATLAB file.
        """
        self.mat_data = loadmat(self.file_path)
        self.keys = list(self.mat_data.keys())
        

    def read_file(self, section='data'):
        """
        Read the specified section of the MATLAB file.

        Args:
            section (str, optional): The section to read from the MATLAB file (default is 'data').

        Returns:
            numpy_array (ndarray or Tensor): The data read from the MATLAB file.
        """
        if section not in self.keys:
            raise ValueError(f'Invalid section {section}.')
        numpy_array = self.mat_data[section]
        if self.to_tensor:
            numpy_array = torch.from_numpy(numpy_array).to(self.device)
        return numpy_array
    
    def __repr__(self):
        return f'MatlabFileReader(file_path={self.file_path}, device={self.device}, to_numpy={self.to_numpy}, to_tensor={self.to_tensor})'
    
    
