import torch
from torch import nn
from util.util import adjust_lr


def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, epochs: int, lr: float, cl=None,
          verbose=False):
    """
    Train the model based on on the train loader
    """
    # Get the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set the model to the device
    model.to(device)
    model.train()

    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_thresh = 7e-4

    # Get the number of epochs for training
    epochs = epochs
    # Loop over each epoch
    for epoch_idx in range(epochs):
        train_loss = 0
        # Loop over the training set
        for data, target in train_loader:
            # Move data to the device
            data, target = data.to(device, non_blocking=True), target.to(
                device, non_blocking=True)
            # Cast target to long tensor
            target = target.long()

            # Set the gradients to zero
            optimizer.zero_grad(set_to_none=True)

            # Get the model output
            output = model(data)

            # Calculate the loss
            loss = criterion(output, target)

            # Perform the backward pass
            loss.backward()
            # Take a step using optimizer
            optimizer.step()
            # if loss.item() < loss_thresh:
            #     loss_thresh = loss_thresh / 10
            #     lr = lr / 5
            #     adjust_lr(optimizer, lr)

            # Add the loss to the total loss
            train_loss += loss.item()

        # Print the epoch and loss summary
        if cl is not None and verbose:
            cl.logger.info(f"Epoch: {epoch_idx + 1}/{epochs} | Loss: {train_loss / len(train_loader):.8f}")
            # torch.save("shadow_model_", os.path.join(cl.log_dir,""))

    return model, optimizer


def com_train(model: torch.nn.Module, com_model: torch.nn.Module, train_loader: torch.utils.data.DataLoader,
              epochs: int, lr: float, cl=None, num_known_features=300,
              verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    com_model.train()

    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(com_model.parameters(), lr=lr)

    # Get the number of epochs for training
    epochs = epochs
    # Loop over each epoch
    for epoch_idx in range(epochs):
        train_loss = 0
        # Loop over the training set
        for data, target in train_loader:
            # Move data to the device
            data, target = data.to(device, non_blocking=True), target.to(
                device, non_blocking=True)
            # Cast target to long tensor
            target = target.long()

            known_features = data[:, :num_known_features]
            unknown_features = data[:, num_known_features:]

            # Set the gradients to zero
            optimizer.zero_grad(set_to_none=True)

            # Get the model output
            out = com_model(known_features)
            com_features = torch.column_stack((known_features, out))

            output = model(com_features)

            # Calculate the loss
            loss = criterion(output, target)

            # Perform the backward pass
            loss.backward()
            # Take a step using optimizer
            optimizer.step()

            # Add the loss to the total loss
            train_loss += loss.item()

        # Print the epoch and loss summary
        if cl is not None and verbose:
            cl.logger.info(f"Epoch: {epoch_idx + 1}/{epochs} | Loss: {train_loss / len(train_loader):.8f}")
            # torch.save("shadow_model_", os.path.join(cl.log_dir,""))

    return com_model, optimizer


def train_AE(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, epochs: int, lr: float, max_dis, lamb,
             cl=None,
             verbose=False):
    """
    Train the model based on on the train loader
    """
    # Get the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set the model to the device
    model.to(device)
    model.train()

    # Set the loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Get the number of epochs for training
    epochs = epochs
    # Loop over each epoch
    for epoch_idx in range(epochs):
        train_loss = 0
        # Loop over the training set
        for data, target in train_loader:
            # Move data to the device
            data, target = data.to(device, non_blocking=True), target.to(
                device, non_blocking=True)
            # Cast target to long tensor
            target = target.long()

            # Set the gradients to zero
            optimizer.zero_grad(set_to_none=True)

            # Get the model output
            enc_out, dec_out = model(data)

            # Calculate the loss
            bs = len(data)
            packet_len = bs // 2
            loss1 = torch.square(torch.norm(data - dec_out, p=2, dim=list(range(1, len(data.shape))))).mean()
            same_class = target[:packet_len] == target[packet_len:]
            left_enc_out = enc_out[:packet_len]
            right_enc_out = enc_out[packet_len:]
            distance = torch.norm(left_enc_out - right_enc_out, p=2, dim=list(range(1, len(left_enc_out.shape))))
            loss2 = torch.sum(torch.square(distance[same_class]))
            loss2 += torch.sum(torch.square(torch.clamp(max_dis - distance[~same_class], min=0)))
            loss = loss1 + lamb * loss2.mean()

            # Perform the backward pass
            loss.backward()
            # Take a step using optimizer
            optimizer.step()

            # Add the loss to the total loss
            train_loss += loss.item()

        # Print the epoch and loss summary
        if cl is not None and verbose:
            cl.logger.info(f"Epoch: {epoch_idx + 1}/{epochs} | Loss: {train_loss / len(train_loader):.8f}")
            # torch.save("shadow_model_", os.path.join(cl.log_dir,""))

    return model, optimizer
