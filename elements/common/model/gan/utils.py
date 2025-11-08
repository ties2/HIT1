import numpy as np
import torch


def generate_latent_space_samples(dims: tuple, device: str = "cuda") -> torch.Tensor:
    """
    Generates samples from a latent space using a normal distribution.

    :param dims: The dimensions of the tensor to be generated, typically including batch size and latent dimension size.
    :param device: The device on which the tensor will be allocated ('cpu' or 'cuda').

    :return: A tensor of random numbers following a normal distribution, allocated on the specified device.
    """
    return torch.randn(dims).to(device=device)


def prepare_real_samples(real_samples, batch_size: int, device: str) -> tuple:
    """
    Prepares real samples and their corresponding labels for training.

    :param real_samples: A batch of real samples from the dataset.
    :param batch_size: The size of the batch of real samples.
    :param device: The device to allocate tensors ('cpu' or 'cuda').

    :return: A tuple of tensors: the real samples and their labels, both moved to the specified device.
             The labels for real samples are set to ones.
    """
    real_samples = real_samples.to(device=device)
    real_samples_labels = torch.ones((batch_size, 1)).to(
        device=device
    )
    return real_samples, real_samples_labels


def prepare_generated_samples(generator, latent_space_samples, batch_size: int, device: str) -> tuple:
    """
    Generates fake samples using the generator and prepares their labels.

    :param generator: The generator network.
    :param latent_space_samples: Random noise vectors from the latent space.
    :param batch_size: The number of fake samples to generate.
    :param device: The device to allocate tensors ('cpu' or 'cuda').

    :return: A tuple of tensors: the generated samples and their labels, both moved to the specified device.
             The labels for generated samples are set to zeros.
    """
    generated_samples = generator(latent_space_samples)
    generated_samples_labels = torch.zeros((batch_size, 1)).to(
        device=device
    )
    return generated_samples, generated_samples_labels


def combine_real_fake_data(real_samples, generated_samples, real_samples_labels, generated_samples_labels) -> tuple:
    """
    Combines real and fake samples along with their labels into single tensors.

    :param real_samples: Tensor containing real samples.
    :param generated_samples: Tensor containing generated (fake) samples.
    :param real_samples_labels: Tensor containing labels for real samples.
    :param generated_samples_labels: Tensor containing labels for generated samples.

    :return: Two tensors: one containing all samples (real and fake) concatenated, and the other containing all labels (real and fake) concatenated.
    """
    all_samples = torch.cat((real_samples, generated_samples))
    all_samples_labels = torch.cat(
        (real_samples_labels, generated_samples_labels)
    )

    return all_samples, all_samples_labels


def reshape_flattened_vector(generated_samples: torch.Tensor, shape: tuple) -> np.ndarray:
    """
    Reshapes a batch of flattened vectors into a batch of 2D images.

    :param generated_samples: A batch of flattened vectors representing images.
    :param shape: The shape to reshape to

    :return: A numpy array of 2D images reshaped from the input flattened vectors.
    """

    generated_samples = np.asarray([sample.reshape(shape).cpu().detach().numpy() for sample in generated_samples])
    return generated_samples
