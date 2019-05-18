import torch
from torch.autograd import Variable
import torch.nn as nn

from fragile.dl.datasets import ImageDataset


class Binarize(nn.Module):
    def forward(self, input):
        input[input >= 0.5] = 1
        input[input < 0.5] = 0
        return input


class Encoder(nn.Module):
    def __init__(self, input_size: int, n_bits: int = 32):
        super(Encoder, self).__init__()
        self.n_bits = n_bits
        self.input_size = input_size
        self.hidden = nn.Linear(input_size, n_bits)
        self.softmax = nn.Softmax()
        self.binarize = Binarize()
        self.encoder_out = None
        self.softmax_out = None
        self.bin_out = None

    def forward(self, input):
        self.encoder_out = self.hidden(input.view(input.shape[0], -1))
        self.softmax_out = self.softmax(self.encoder_out)
        self.binarize(self.softmax_out)
        return self.softmax_out

    def binarize(self, input):
        self.bin_out = self.binarize(input)
        return self.bin_out


class Decoder(nn.Module):
    def __init__(self, input_size: int, n_bits: int = 32):
        super(Decoder, self).__init__()
        self.n_bits = n_bits
        self.input_size = input_size
        self.hidden = nn.Linear(n_bits, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.hidden(input)
        return self.sigmoid(x)


class AutoEncoder(nn.Module):
    def __init__(self, input_size: int, n_bits: int = 32):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_size=input_size, n_bits=n_bits)
        self.decoder = Decoder(input_size=input_size, n_bits=n_bits)

    def forward(self, input):
        x = self.encoder(input)
        return self.decoder(x)


class Mapper:
    def __init__(self, dataset: ImageDataset, input_size: int, n_bits: int = 32):
        self.dataset = dataset
        self.autoencoder = AutoEncoder(input_size=input_size, n_bits=n_bits)
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), weight_decay=1e-5)
        self.distance = nn.MSELoss()

    def calculate_loss(self, output, target):
        l1_weight = 0.01
        distance_weight = 0.1
        entropy_weight = 1.0

        bin_vectors = self.autoencoder.encoder.bin_out
        distance = self.distance(output, target)
        by_columns = bin_vectors.mean(0)
        entropy = by_columns * torch.log(by_columns)
        entropy = -1 * entropy.mean()
        loss = (
            distance_weight * distance - entropy_weight * entropy + l1_weight * bin_vectors.sum(1)
        )
        return loss

    def train_autoencoder(self, num_epochs: int = 10):
        for epoch in range(num_epochs):
            for data in self.dataset.dataloader:
                img, _ = data
            img = Variable(img).cpu()
            # ===================forward=====================
            output = self.autoencoder(img)
            loss = self.calculate_loss(output, img)
            # ===================backward====================
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                print("epoch [{}/{}], loss:{:.4f}".format(epoch + 1, num_epochs, loss.data))
        # ===================log========================
        print("epoch [{}/{}], loss:{:.4f}".format(epoch + 1, num_epochs, loss.data))
